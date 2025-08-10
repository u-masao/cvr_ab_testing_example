import io
import textwrap

import arviz as az
import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pymc as pm
from PIL import Image

from src.visualization.visualize import plot_distribution


def define_model(trials: list[int], successes: list[int]) -> pm.Model:
    """モデル定義"""

    # 入力チェック
    if len(trials) != len(successes):
        raise ValueError("試行回数と成功回数のグループ数が一致しません")

    shape = len(trials)
    model = pm.Model()

    with model:
        # 二項分布のパラメーターをベータ分布で定義(無情報)
        p = pm.Beta("p", alpha=1.0, beta=1.0, shape=shape)

        # 二項分布のデータとパラメーターを定義
        obs = pm.Binomial(  # noqa: F841
            "y", n=trials, observed=successes, p=p, shape=shape
        )

        # アップリフト率を定義(決定論的)
        relative_uplift = pm.Deterministic(  # noqa: F841
            "relative_uplift", p[1] / p[0] - 1.0
        )

        # アップリフトを定義(決定論的)
        uplift = pm.Deterministic("uplift", p[1] - p[0])  # noqa: F841

    return model


def analyze(model: pm.Model, trace, hdi_prob: float) -> tuple:
    """結果を分析して可視化"""
    # distribution
    dist_plot = plot_distribution(None, None, trace, None)

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


def simulate(
    n_a,
    conversion_a,
    n_b,
    conversion_b,
    n_sampling,
    n_tune,
    n_chains,
    random_seed,
    hdi_prob,
):
    """シミュレーション実行"""

    # モデル定義
    model = define_model([n_a, n_b], [conversion_a, conversion_b])

    with model:
        # サンプリング
        trace = pm.sample(
            n_sampling,
            tune=n_tune,
            chains=n_chains,
            random_seed=random_seed,
        )

    # 分析と可視化
    return analyze(model, trace, hdi_prob)


with gr.Blocks() as demo:
    header = gr.Markdown(
        textwrap.dedent(
            """
                # ベイズ推定による A/B テストの分析
            """
        )
    )

    with gr.Row():
        with gr.Column():
            n_a = gr.Number(100, label="A 群の試行回数")
            conversion_a = gr.Number(4, label="A 群のコンバージョン数")
        with gr.Column():
            n_b = gr.Number(1000, label="B 群の試行回数")
            conversion_b = gr.Number(55, label="B 群のコンバージョン数")
    with gr.Accordion("ベイズ推定の詳細設定", open=False):
        n_sampling = gr.Number(4000, label="サンプリング回数")
        n_tune = gr.Number(1000, label="バーンイン回数")
        n_chains = gr.Number(4, label="チェーン数")
        random_seed = gr.Number(1234, label="ランダムシード")
        hdi_prob = gr.Number(0.95, label="Highest Density Interval")

    start_button = gr.Button("推定開始")

    with gr.Group():
        gr.Markdown("## 結果")
        distribution = gr.Plot()
        gr.Markdown("## モデル")
        model = gr.Image()
        gr.Markdown("## トレースプロットによる収束確認")
        trace_plot = gr.Plot()
        gr.Markdown("## r_hat による収束確認")
        params = gr.Plot()

    start_button.click(
        simulate,
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
        outputs=[distribution, model, trace_plot, params],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
    )
