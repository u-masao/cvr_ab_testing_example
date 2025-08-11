import io
from typing import Dict, Union

import arviz as az
import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop
from PIL import Image
from scipy.stats import chi2_contingency
from tqdm import tqdm

from src.models.base import define_model
from src.visualization.visualize import plot_distribution


def calculate_sample_size(
    baseline_cvr: float,
    lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> Dict[str, Union[float, int]]:
    """
    A/BテストのCVRに基づいて必要なサンプルサイズを計算し、結果を辞書で返します。

    Args:
        baseline_cvr (float): ベースラインのCVR（例: 0.03）。
        lift (float): 検出したいCVRの絶対的な改善量（例: 0.005）。
        alpha (float, optional): 有意水準。デフォルトは 0.05。
        power (float, optional): 検出力。デフォルトは 0.80。
        ratio (float, optional): グループBのサンプルサイズ / グループAのサンプルサイズの比率。
                                 デフォルトは 1.0 (均等)。

    Returns:
        Dict[str, Union[float, int]]: 計算結果を含む辞書。
            - baseline_cvr: 入力されたベースラインCVR
            - target_cvr: 目標CVR (baseline_cvr + lift)
            - lift: 入力された改善量
            - alpha: 有意水準
            - power: 検出力
            - ratio: グループ間のサンプルサイズ比率
            - effect_size: 計算された効果量 (Cohen's h)
            - required_sample_size_group_a: グループAに必要なサンプルサイズ
            - required_sample_size_group_b: グループBに必要なサンプルサイズ
            - total_sample_size: 合計サンプルサイズ
    """
    # 効果量（Effect Size）を計算します。
    effect_size = smprop.proportion_effectsize(
        baseline_cvr, baseline_cvr + lift
    )

    # グループAに必要なサンプルサイズ（nobs1）を計算します。
    # ratioは nobs2/nobs1 として扱われます。
    required_nobs1 = smp.NormalIndPower().solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative="two-sided",  # 両側検定
    )

    # 計算結果を整数に切り上げます。
    nobs1 = int(np.ceil(required_nobs1))
    nobs2 = int(np.ceil(nobs1 * ratio))

    # 結果を辞書にまとめる
    result_dict = {
        "baseline_cvr": baseline_cvr,
        "target_cvr": baseline_cvr + lift,
        "lift": lift,
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
        "effect_size": effect_size,
        "required_sample_size_group_a": nobs1,
        "required_sample_size_group_b": nobs2,
        "total_sample_size": nobs1 + nobs2,
    }

    return result_dict


def run_chisquared_test(n_a, conversion_a, n_b, conversion_b) -> pd.DataFrame:
    """
    A/Bテストデータに対してカイ二乗検定を実行します。

    Parameters
    ----------
    n_a, conversion_a : int
        A群の試行回数とコンバージョン数。
    n_b, conversion_b : int
        B群の試行回数とコンバージョン数。

    Returns
    -------
    pd.DataFrame
        カイ二乗検定の結果（カイ二乗値、p値、自由度）を格納したDataFrame。
    """
    # 2x2の分割表を作成
    #        | Conversion | No Conversion |
    # -------|------------|---------------|
    # Group A| conv_a     | n_a - conv_a  |
    # Group B| conv_b     | n_b - conv_b  |
    table = np.array(
        [
            [conversion_a, n_a - conversion_a],
            [conversion_b, n_b - conversion_b],
        ]
    )

    chi2, p, dof, _ = chi2_contingency(table, correction=False)

    return pd.DataFrame(
        {
            "指標": ["カイ二乗値", "p値", "自由度"],
            "値": [f"{chi2:.4f}", f"{p:.4f}", dof],
        }
    )


def analyze_bayesian_results(model: pm.Model, trace, hdi_prob: float) -> tuple:
    """
    ベイズ分析のサンプリング結果を分析し、可視化のためのプロットを生成します。

    Parameters
    ----------
    model : pm.Model
        PyMCモデルオブジェクト。
    trace : az.InferenceData
        サンプリング結果のトレース。
    hdi_prob : float
        最高事後密度区間（HDI）の確率。

    Returns
    -------
    tuple
        可視化されたプロットのタプル:
        (分布プロット, モデル構造, トレースプロット, フォレストプロット)
    """
    # UIからは真の値や観測値の平均はプロットしないため、最小限のdata辞書を作成
    data_for_plot = {"p_a_true": None, "p_b_true": None}
    dist_plot = plot_distribution(data_for_plot, trace)

    # model arch
    png_data = pm.model_to_graphviz(model).pipe(format="png")
    model_arch = Image.open(io.BytesIO(png_data))

    # trace plot
    trace_plot, axes = plt.subplots(4, 2, figsize=(12, 8))
    pm.plot_trace(trace, compact=False, axes=axes)
    trace_plot.tight_layout()
    for ax in axes.flatten():
        ax.grid()

    # forest
    var_names = trace.posterior.data_vars.keys()
    var_length = len(var_names)
    forest, axes = plt.subplots(
        var_length,
        2,
        figsize=(12, var_length * 3),
    )
    for index, var_name in enumerate(var_names):
        az.plot_forest(
            trace,
            var_names=[var_name],
            combined=True,
            r_hat=True,
            hdi_prob=hdi_prob,
            ax=axes[index],
        )
    forest.tight_layout()
    for ax in axes.flatten():
        ax.grid()

    return dist_plot, model_arch, trace_plot, forest


def run_bayesian_analysis(
    n_a,
    conversion_a,
    n_b,
    conversion_b,
    n_sampling,
    n_tune,
    n_chains,
    random_seed,
    hdi_prob,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio UIからの入力に基づいて、A/Bテストのベイズ分析を実行します。

    Parameters
    ----------
    (省略)

    Returns
    -------
    tuple
        analyze_bayesian_results関数から返されるプロットのタプル。
    """

    # モデル定義
    model = define_model([n_a, n_b], [conversion_a, conversion_b])

    with model:
        # プログレス設定
        pbar = tqdm(total=n_chains * (n_sampling + n_tune), desc="サンプリング中")

        # プログレス更新
        def update_progress(trace, draw):
            pbar.update(1)

        # サンプリング
        trace = pm.sample(
            draws=n_sampling,
            tune=n_tune,
            chains=n_chains,
            random_seed=random_seed,
            callback=update_progress,
        )

        # プログレスクローズ
        pbar.close()

    # 分析と可視化
    return analyze_bayesian_results(model, trace, hdi_prob)


def run_power_analysis(baseline_cvr, lift, alpha, power, ratio):
    test_params = {
        "baseline_cvr": baseline_cvr,
        "lift": lift,
        "alpha": alpha,
        "power": power,
        "ratio": ratio,  # 例: グループAとBを1:1にする場合は1.0、A:B=1:2にする場合は2.0
    }
    result = calculate_sample_size(**test_params)
    result_df = pd.DataFrame([result]).T.reset_index()
    result_df.columns = ["指標", "値"]
    return (
        result["required_sample_size_group_a"],
        result["required_sample_size_group_b"],
        result_df,
    )


text_about_this_tool = """
このツールは、A/Bテストの結果を「ベイズ統計」と「頻度論統計（カイ二乗検定）」の2つのアプローチで分析し、比較することができます。
- **ベイズ分析**: 施策の優劣を「確率」として解釈し、柔軟な意思決定をサポートします。
- **カイ二乗検定**: 古典的な統計手法で、2つのグループ間に「統計的に有意な差」があるかどうかをp値で判断します。
"""

text_about_baysian_analysis = """
#### 分布プロットの見方
- **p_a, p_bの分布**: それぞれのグループのコンバージョン率が、どの値を取りそう
かという「確信度」を分布で表しています。幅が狭いほど、推定の確信度が高いことを意味します。
- **uplift (p_b - p_a)の分布**: BがAよりどれだけ良いか（絶対差）の分布です。
    - 分布全体が0より大きければ、「BはAより良い」と強く言えそうです。
    - 0をまたいでいる場合、優劣の判断は不確実です。分布が0より大きい部分の面積が「BがAを上回る確率」に相当します。
- **relative_uplift ((p_b - p_a)/p_a)の分布**: 改善率の分布です。「CVRが何%改善したか」を確率的に解釈できます。
#### こんな時に便利
- 「BがAより良い確率は？」といった問いに直接答えたい時。
- テスト期間が短く、サンプル数が少なくても、暫定的な示唆を得たい時。
- 「有意差なし」という結果だけでなく、優劣の確率を知り、ビジネス判断に活かしたい時。
"""

text_about_chi_squared_test = """
#### p値の解釈
- **p値**は、「もしAとBの間に本当は差がないとしたら、今回観測されたような差（またはそれ以上の差）が偶然生じる確率」を示します。
- 慣習的に、p値が0.05（5%）を下回る場合に「統計的に有意な差がある」と判断し、「AとBのコンバージョン率には差がある」と結論付けます。
- p値が0.05を上回る場合は、「統計的に有意な差があるとは言えない」と結論付けます。これは「差がない」ことを証明するものではない点に注意が必要です。
#### こんな時に便利
- 伝統的な統計的仮説検定の枠組みで、客観的な「有意差」の有無を判断したい時。
- レポートなどで、広く受け入れられているp値を報告する必要がある時。
#### 注意点: pハッキングと検定の恣意的な停止
- **pハッキング**: p値が0.05を下回るまで、分析方法を色々試したり、データを追加
したり、逆に一部を除外したりする行為は「pハッキング」と呼ばれ、誤った結論を導く原因となります。
- **恣意的な検定停止**: テストの途中でp値が0.05を下回ったからといって、そこでテストを終了してはいけません。
これは「偽陽性（本当は差がないのに、差があると判断してしまうこと）」の確率を高める行為です。テストは、
事前に決めたサンプルサイズに達するまで続ける必要があります。
"""

text_about_power_analysis = """
**検出力分析 (Power Analysis)** は、A/Bテストを**実施する前**に行う、適切なサンプルサイズを見積もるための統計的な手法です。

#### なぜ必要か？
- サンプルサイズが小さすぎると、本当に効果のある施策でも「差がない」という誤った結論に至るリスク（偽陰性）が高まります。
- 逆に、サンプルサイズが大きすぎると、実務的には意味のないごく僅かな差でも「統計的に有意」という結果になりやすくなる上、
時間やコストも無駄にかかってしまいます。検出力分析は、これらのリスクとコストのバランスを取るために役立ちます。

#### 主要な4つの要素
1.  **検出力 (Power, 1-β)**: 本当に差がある時に、それを「差がある」と正しく検出できる確率。一般的に80% (0.8) に設定されます。
2.  **有意水準 (Significance Level, α)**: 本当は差がないのに、「差がある」と間違って判断してしまう確率（偽陽性）。
通常、5% (0.05) に設定されます。
3.  **効果量 (Effect Size)**: 検出したい差の大きさ（例: CVRが2%から2.5%に改善）。効果量が小さいほど、
それを検出するためにより多くのサンプルサイズが必要になります。
4.  **サンプルサイズ (Sample Size)**: 各グループのユーザー数や試行回数。

このツールでは検出力分析は行えませんが、A/Bテストを計画する際には、これらの概念を理解し、専用の計算ツールなどを使って事前にサンプルサイズを見積もることが非常に重要です。
"""

text_bayesien_vs_chi_squared = """
| 特徴 | ベイズ分析 | カイ二乗検定 |
|:---|:---|:---|
| **得られる結果** | パラメータの確率分布 | p値 |
| **解釈の仕方** | 「BがAより良い確率はX%」 | 「有意差がある/ない」 |
| **サンプルサイズ**| 小さくても示唆は得られる | ある程度の大きさが必要 |
| **柔軟性** | 事前知識をモデルに組込可能 | 困難 |
| **計算コスト** | 高い（MCMCサンプリング） | 非常に低い |

**使い分けのヒント**:
- **カイ二乗検定**で「そもそも差があると言えるのか？」を素早く確認し、
- **ベイズ分析**で「では、どのくらい差がありそうか？」「Bを選ぶべき確率は？」といった、より踏み込んだ意思決定を行う、という使い分けが有効です。
"""


# Gradio UIの定義
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# A/Bテスト分析ツール (ベイズ推定 vs カイ二乗検定)")

    with gr.Accordion("❓ このツールで何ができるか", open=False):
        gr.Markdown(text_about_this_tool)

    # --- 入力コンポーネント ---
    gr.Markdown("## 1. テスト結果の入力")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### A群")
            n_a = gr.Number(1000, label="試行回数 (サンプルサイズ)")
            conversion_a = gr.Number(53, label="コンバージョン数")
        with gr.Column():
            gr.Markdown("### B群")
            n_b = gr.Number(100, label="試行回数 (サンプルサイズ)")
            conversion_b = gr.Number(10, label="コンバージョン数")

    with gr.Accordion("⚙️ ベイズ分析の詳細設定", open=False):
        n_sampling = gr.Number(4000, label="サンプリング回数 (draws)")
        n_tune = gr.Number(1000, label="チューニング回数 (tune)")
        n_chains = gr.Number(4, label="チェーン数")
        hdi_prob = gr.Number(
            0.95, label="Highest Density Interval (HDI:最高事後密度区間)"
        )
        random_seed = gr.Number(1234, label="ランダムシード")

    start_button = gr.Button("分析開始", variant="primary")

    # --- 出力コンポーネント ---
    gr.Markdown("## 2. 分析結果")
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ベイズ分析の結果")
            with gr.Tabs():
                with gr.TabItem("📈 分布"):
                    distribution = gr.Plot()
                with gr.TabItem("📊 サマリー"):
                    params = gr.Plot()
                with gr.TabItem("🕸️ モデル"):
                    model_img = gr.Image()
                with gr.TabItem("🛰️ トレースプロット"):
                    trace_plot = gr.Plot()
        with gr.Column(scale=1):
            gr.Markdown("### カイ二乗検定の結果")
            chisq_result_df = gr.DataFrame(
                headers=["指標", "値"], datatype=["str", "str"], label="検定結果"
            )

    # --- 解説セクション ---
    with gr.Accordion("💡 結果の解釈と解説", open=False):
        with gr.Tabs():
            with gr.TabItem("ベイズ分析の解説"):
                gr.Markdown(text_about_baysian_analysis)
            with gr.TabItem("カイ二乗検定の解説"):
                gr.Markdown(text_about_chi_squared_test)
            with gr.TabItem("検出力分析とは？"):
                gr.Markdown(text_about_power_analysis)
            with gr.TabItem("ベイズ vs カイ二乗"):
                gr.Markdown(text_bayesien_vs_chi_squared)
    with gr.Accordion("🔢 サンプルサイズの計算", open=False):
        with gr.Row():
            with gr.Column():
                power_analysis_inputs = {
                    "baseline_cvr": gr.Number(
                        0.053,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        label="ベースラインCVR",
                    ),
                    "lift": gr.Number(
                        0.01,
                        minimum=-1.0,
                        maximum=1.0,
                        step=0.001,
                        label="検出したいCVRの増加量",
                    ),
                    "alpha": gr.Number(
                        0.05,
                        minimum=0.01,
                        maximum=0.5,
                        step=0.01,
                        label="有意水準(Alpha)",
                    ),
                    "power": gr.Number(
                        0.80,
                        minimum=0.5,
                        maximum=0.99,
                        step=0.01,
                        label="検出力(1-Beta)",
                    ),
                    "ratio": gr.Number(
                        0.10,
                        minimum=0.0,
                        maximum=1000,
                        step=0.01,
                        label="A群に対するB群のサンプルサイズ比",
                    ),
                }
            with gr.Column():
                power_analysis_sample_size_a = gr.Number(label="A 群のサンプルサイズ")
                power_analysis_sample_size_b = gr.Number(label="B 群のサンプルサイズ")
                with gr.Accordion("分析結果の詳細", open=False):
                    gr.Markdown("Cohen's h で効果量を計算し、サンプルサイズを計算しています")
                    power_analysis_output = gr.DataFrame()

    # --- イベントリスナー ---
    chisq_event = start_button.click(
        fn=run_chisquared_test,
        inputs=[
            n_a,
            conversion_a,
            n_b,
            conversion_b,
        ],
        outputs=[
            chisq_result_df,
        ],
    )
    chisq_event.then(
        fn=run_bayesian_analysis,
        inputs=[
            n_a,
            conversion_a,
            n_b,
            conversion_b,
            n_sampling,
            n_tune,
            n_chains,
            random_seed,
            hdi_prob,
        ],
        outputs=[
            distribution,
            model_img,
            trace_plot,
            params,
        ],
    )
    gr.on(
        triggers=[x.change for x in power_analysis_inputs.values()],
        fn=run_power_analysis,
        inputs=[x for x in power_analysis_inputs.values()],
        outputs=[
            power_analysis_sample_size_a,
            power_analysis_sample_size_b,
            power_analysis_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
    )
