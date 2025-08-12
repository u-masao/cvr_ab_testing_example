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
