import io
import textwrap

import arviz as az
import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from PIL import Image
from scipy.stats import chi2_contingency

from src.models.base import define_model
from src.visualization.visualize import plot_distribution


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
    table = np.array([
        [conversion_a, n_a - conversion_a],
        [conversion_b, n_b - conversion_b],
    ])

    chi2, p, dof, _ = chi2_contingency(table, correction=False)

    return pd.DataFrame({
        "指標": ["カイ二乗値", "p値", "自由度"],
        "値": [f"{chi2:.4f}", f"{p:.4f}", dof],
    })


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
    dist_plot = plot_distribution(data_for_plot, trace, None)

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
        var_length, 2, sharey=True, figsize=(12, var_length * 3)
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
        # サンプリング
        trace = pm.sample(
            draws=n_sampling,
            tune=n_tune,
            chains=n_chains,
            random_seed=random_seed,
            progressbar=True,
        )

    # 分析と可視化
    return analyze_bayesian_results(model, trace, hdi_prob)


def run_analysis(
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
    UIから呼び出されるメインの分析関数。
    ベイズ分析とカイ二乗検定の両方を実行します。
    """
    bayesian_plots = run_bayesian_analysis(
        n_a,
        conversion_a,
        n_b,
        conversion_b,
        n_sampling,
        n_tune,
        n_chains,
        random_seed,
        hdi_prob,
        progress,
    )
    chisq_results = run_chisquared_test(n_a, conversion_a, n_b, conversion_b)

    # gr.update()を使って各コンポーネントを更新
    return (
        gr.update(value=bayesian_plots[0]),
        gr.update(value=bayesian_plots[1]),
        gr.update(value=bayesian_plots[2]),
        gr.update(value=bayesian_plots[3]),
        gr.update(value=chisq_results),
    )


# Gradio UIの定義
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# A/Bテスト分析ツール (ベイズ推定 vs カイ二乗検定)")

    with gr.Accordion("❓ このツールで何ができるか", open=False):
        gr.Markdown(
            """
            このツールは、A/Bテストの結果を「ベイズ統計」と「頻度論統計（カイ二乗検定）」の2つのアプローチで分析し、比較することができます。
            - **ベイズ分析**: 施策の優劣を「確率」として解釈し、柔軟な意思決定をサポートします。
            - **カイ二乗検定**: 古典的な統計手法で、2つのグループ間に「統計的に有意な差」があるかどうかをp値で判断します。
            """
        )

    # --- 入力コンポーネント ---
    gr.Markdown("## 1. テスト結果の入力")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### A群")
            n_a = gr.Number(100, label="試行回数 (サンプルサイズ)")
            conversion_a = gr.Number(4, label="コンバージョン数")
        with gr.Column():
            gr.Markdown("### B群")
            n_b = gr.Number(1000, label="試行回数 (サンプルサイズ)")
            conversion_b = gr.Number(55, label="コンバージョン数")

    with gr.Accordion("⚙️ ベイズ分析の詳細設定", open=False):
        n_sampling = gr.Number(4000, label="サンプリング回数 (draws)")
        n_tune = gr.Number(1000, label="チューニング回数 (tune)")
        n_chains = gr.Number(4, label="チェーン数")
        random_seed = gr.Number(1234, label="ランダムシード")
        hdi_prob = gr.Number(0.95, label="Highest Density Interval (HDI)")

    start_button = gr.Button("分析開始", variant="primary")

    # --- 出力コンポーネント ---
    gr.Markdown("## 2. 分析結果")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ベイズ分析の結果")
            with gr.Tabs():
                with gr.TabItem("📈 分布プロット"):
                    distribution = gr.Plot()
                with gr.TabItem("📊 サマリープロット"):
                    params = gr.Plot()
                with gr.TabItem("🕸️ モデル構造"):
                    model_img = gr.Image()
                with gr.TabItem("🛰️ トレースプロット (収束確認)"):
                    trace_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("### カイ二乗検定の結果")
            chisq_result_df = gr.DataFrame(
                headers=["指標", "値"], datatype=["str", "str"], label="検定結果"
            )

    # --- 解説セクション ---
    with gr.Accordion("💡 結果の解釈と解説", open=False):
        with gr.Tabs():
            with gr.TabItem("ベイズ分析の解説"):
                gr.Markdown(
                    """
                    #### 分布プロットの見方
                    - **p_a, p_bの分布**: それぞれのグループのコンバージョン率が、どの値を取りそうかという「確信度」を分布で表しています。幅が狭いほど、推定の確信度が高いことを意味します。
                    - **uplift (p_b - p_a)の分布**: BがAよりどれだけ良いか（絶対差）の分布です。
                        - 分布全体が0より大きければ、「BはAより良い」と強く言えそうです。
                        - 0をまたいでいる場合、優劣の判断は不確実です。分布が0より大きい部分の面積が「BがAを上回る確率」に相当します。
                    - **relative_uplift ((p_b - p_a)/p_a)の分布**: 改善率の分布です。「CVRが何%改善したか」を確率的に解釈できます。
                    #### こんな時に便利
                    - 「BがAより良い確率は？」といった問いに直接答えたい時。
                    - テスト期間が短く、サンプル数が少なくても、暫定的な示唆を得たい時。
                    - 「有意差なし」という結果だけでなく、優劣の確率を知り、ビジネス判断に活かしたい時。
                    """
                )
            with gr.TabItem("カイ二乗検定の解説"):
                gr.Markdown(
                    """
                    #### p値の解釈
                    - **p値**は、「もしAとBの間に本当は差がないとしたら、今回観測されたような差（またはそれ以上の差）が偶然生じる確率」を示します。
                    - 慣習的に、p値が0.05（5%）を下回る場合に「統計的に有意な差がある」と判断し、「AとBのコンバージョン率には差がある」と結論付けます。
                    - p値が0.05を上回る場合は、「統計的に有意な差があるとは言えない」と結論付けます。これは「差がない」ことを証明するものではない点に注意が必要です。
                    #### こんな時に便利
                    - 伝統的な統計的仮説検定の枠組みで、客観的な「有意差」の有無を判断したい時。
                    - レポートなどで、広く受け入れられているp値を報告する必要がある時。
                    """
                )
            with gr.TabItem("ベイズ vs カイ二乗"):
                gr.Markdown(
                    """
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
                )

    # --- イベントリスナー ---
    start_button.click(
        run_analysis,
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
            chisq_result_df,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
    )
