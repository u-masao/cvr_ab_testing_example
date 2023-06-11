import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import click
import cloudpickle
import japanize_matplotlib  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pymc as pm

from src.utils import make_fig_from_axes, savefig


def calc_summary_of_obs_and_true(observations: List, p_true: float) -> Dict:
    """
    観測値のサマリーと真値との違いを計算
    """
    disp_length = np.min([20, len(observations)])
    return {
        "obs": observations[:disp_length],
        "obs_len": len(observations),
        "obs_sum": np.sum(observations),
        "obs_mean": np.mean(observations),
        "obs_mean - p_true:": np.mean(observations) - p_true,
        "(obs_mean - p_true) / p_true:": (np.mean(observations) - p_true)
        / p_true,
    }


def plot_histogram_single(
    ax,
    p_true,
    sample,
    value_name="",
    color_number=None,
    hdi_prob=0.95,
    cumulative=False,
) -> None:
    """
    ヒストグラムを描画
    """
    # 描画色を指定
    if color_number is not None:
        color = plt.get_cmap("Dark2")(color_number % 10)
    else:
        color = plt.get_cmap("Dark2")(int(np.random.rand() * 10))

    # 確信区間を計算
    ci_low, ci_high = az.hdi(sample, hdi_prob=hdi_prob)

    # タイトルを指定
    ax.set_title(f"{value_name} の分布")

    # ヒストグラムのオプションを指定
    hist_args = dict(
        bins=25,
        label=f"{value_name} の分布",
        alpha=0.4,
        color=color,
        density=True,
    )

    # 累積分布関数を表示する際のオプションを指定
    if cumulative:
        hist_args["cumulative"] = True
        hist_args["histtype"] = "step"
        hist_args["alpha"] = 0.9
        hist_args["bins"] = 250

    # plot
    n, _, _ = ax.hist(sample, **hist_args)
    ax.vlines(
        p_true,
        0,
        np.max(n) * 1.2,
        linestyle="--",
        label=f"{value_name} の真の値",
        colors=[color],
        alpha=0.5,
    )
    ci_bar_y = np.max(n) * (0.1 + 0.2 * np.random.rand())
    ax.plot(
        [ci_low, ci_high],
        [ci_bar_y] * 2,
        linestyle="-",
        label=f"{hdi_prob * 100:0.0f} 確信区間 {value_name}",
        color=color,
        alpha=0.9,
        marker="x",
    )
    ax.annotate(
        f"{ci_low:0.3f}",
        xy=(ci_low, ci_bar_y + 0.05 * np.max(n)),
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"{ci_high:0.3f}",
        xy=(ci_high, ci_bar_y + 0.05 * np.max(n)),
        ha="center",
        va="bottom",
    )


def plot_histogram_overlap(
    ax, p_a_true, p_b_true, burned_trace, cumulative=False
) -> None:
    """
    オーバーラップしたヒストグラムを描画
    """
    p_a = burned_trace.posterior["p"][:, :, 0].values.flatten()
    p_b = burned_trace.posterior["p"][:, :, 1].values.flatten()
    plot_histogram_single(
        ax,
        p_a_true,
        p_a,
        value_name="$p_a$",
        color_number=0,
        cumulative=cumulative,
    )
    plot_histogram_single(
        ax,
        p_b_true,
        p_b,
        value_name="$p_b$",
        color_number=2,
        cumulative=cumulative,
    )
    ax.set_title("$p_a$ と $p_b$ の分布")


def plot_distribution(p_a_true, p_b_true, trace) -> mpl.figure.Figure:
    """plot histogram"""
    fig, axes = plt.subplots(5, 2, figsize=(16, 12))
    axes = axes.flatten()

    p_a = trace.posterior["p"][:, :, 0].values.flatten()
    p_b = trace.posterior["p"][:, :, 1].values.flatten()

    for offset, cumulative in enumerate([False, True]):
        options = {"cumulative": cumulative}
        plot_histogram_overlap(
            axes[0 + offset], p_a_true, p_b_true, trace, **options
        )
        plot_histogram_single(
            axes[2 + offset],
            p_a_true,
            p_a,
            value_name="$p_a$",
            color_number=2,
            **options,
        )
        plot_histogram_single(
            axes[4 + offset],
            p_b_true,
            p_b,
            value_name="$p_b$",
            color_number=4,
            **options,
        )
        plot_histogram_single(
            axes[6 + offset],
            p_b_true - p_a_true,
            p_b - p_a,
            value_name="$p_b - p_a$",
            color_number=6,
            **options,
        )
        plot_histogram_single(
            axes[8 + offset],
            (p_b_true - p_a_true) / p_a_true,
            (p_b - p_a) / p_a,
            value_name="$(p_b - p_a) / p_a$",
            color_number=8,
            **options,
        )

    # 軸のスケールを一致
    for reference in [0, 1]:
        axes[1 * 2 + reference].sharex(axes[reference])
        axes[2 * 2 + reference].sharex(axes[reference])

    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid()

    fig.suptitle("$p_A$ と $p_B$ の事後分布と真の値")
    fig.tight_layout()
    return fig


def load_theta(filepath) -> Tuple:
    """load true parameters"""
    theta = pickle.load(open(filepath, "rb"))
    p_a_true = theta["p_a_true"]
    p_b_true = theta["p_b_true"]
    n_a = theta["n_a"]
    n_b = theta["n_b"]
    observations_a = theta["obs_a"]
    observations_b = theta["obs_b"]
    return p_a_true, p_b_true, n_a, n_b, observations_a, observations_b


def calc_ci(posterior, hdi_prob=0.95) -> Dict:
    # init log
    logger = logging.getLogger(__name__)

    # pickup sample
    p_a = posterior["p"][:, :, 0].values.flatten()
    p_b = posterior["p"][:, :, 1].values.flatten()
    p_diff = posterior["uplift"].values.flatten()
    p_ratio = posterior["relative_uplift"].values.flatten()

    # 確信区間を計算
    p_a_ci_low, p_a_ci_high = az.hdi(p_a, hdi_prob=hdi_prob)
    p_b_ci_low, p_b_ci_high = az.hdi(p_b, hdi_prob=hdi_prob)
    p_diff_ci_low, p_diff_ci_high = az.hdi(p_diff, hdi_prob=hdi_prob)
    p_ratio_ci_low, p_ratio_ci_high = az.hdi(p_ratio, hdi_prob=hdi_prob)
    ci = {
        "p_a": {"ci_low": p_a_ci_low, "ci_high": p_a_ci_high},
        "p_b": {"ci_low": p_b_ci_low, "ci_high": p_b_ci_high},
        "p_diff": {"ci_low": p_diff_ci_low, "ci_high": p_diff_ci_high},
        "p_ratio": {"ci_low": p_ratio_ci_low, "ci_high": p_ratio_ci_high},
    }
    logger.info(f"確信区間: {ci}")
    return ci


def calc_prob_dist(samples, hdi_prob=0.95, divide=100) -> pd.DataFrame:
    """
    累積分布を計算
    """
    ci_low, ci_high = az.hdi(samples, hdi_prob=hdi_prob)
    values = np.linspace(ci_low, ci_high, divide)
    prob = [(samples < x).mean() for x in values]
    return pd.DataFrame({"value": values, "prob": prob})


def calc_prob_for_dicision(
    trace,
    model,
    p_a_true: float,
    p_b_true: float,
    observations_a: List,
    observations_b: List,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    # 評価対象を作成
    p_a = trace.posterior["p"][:, :, 0].values.flatten()
    p_b = trace.posterior["p"][:, :, 1].values.flatten()
    p_diff = p_b - p_a
    p_ratio = p_diff / p_a

    # 差と確率
    results = []
    for value, name in [
        [p_a, "p_a"],
        [p_b, "p_b"],
        [p_diff, "p_diff"],
        [p_ratio, "p_ratio"],
    ]:
        results.append(calc_prob_dist(value).assign(name=name))
    return pd.concat(results)


def save_csv_and_log_artifact(df, path):
    df.to_csv(path)
    mlflow.log_artifact(path)


def output_results(
    p_a_true,
    p_b_true,
    model,
    trace,
    metrics,
    hdi_prob,
    prob_summary_df,
    kwargs,
) -> None:
    """結果を出力する"""

    # make dirs
    os.makedirs(kwargs["csv_output_dir"], exist_ok=True)
    os.makedirs(kwargs["figure_dir"], exist_ok=True)

    # 指標を出力
    save_csv_and_log_artifact(
        pd.DataFrame(metrics), Path(kwargs["csv_output_dir"]) / "metrics.csv"
    )

    # summary を出力
    save_csv_and_log_artifact(
        az.summary(trace, round_to=None, hdi_prob=hdi_prob),
        Path(kwargs["csv_output_dir"]) / "sampling_summary.csv",
    )

    # 意思決定に利用する確率を出力
    save_csv_and_log_artifact(
        prob_summary_df,
        Path(kwargs["csv_output_dir"]) / "prob_for_dicision.csv",
    )

    # 各 chain のトレースプロットを出力
    savefig(
        make_fig_from_axes(
            axes=pm.plot_trace(trace, compact=False, combined=False)
        ),
        Path(kwargs["figure_dir"]) / "traceplot.png",
        mlflow_log_artifact=True,
    )

    # サンプルの分布を出力
    savefig(
        plot_distribution(
            p_a_true,
            p_b_true,
            trace,
        ),
        Path(kwargs["figure_dir"]) / "distribution.png",
        mlflow_log_artifact=True,
    )

    # 事後分布を出力
    savefig(
        make_fig_from_axes(
            az.plot_posterior(trace, hdi_prob=hdi_prob),
        ),
        Path(kwargs["figure_dir"]) / "posterior.png",
        mlflow_log_artifact=True,
    )

    # プロット
    savefig(
        make_fig_from_axes(
            az.plot_dist(trace.prior_predictive["relative_uplift"]),
        ),
        Path(kwargs["figure_dir"]) / "distribution_with_obs.png",
        mlflow_log_artifact=True,
    )

    # forest を出力
    savefig(
        make_fig_from_axes(
            az.plot_forest(
                trace, combined=True, hdi_prob=hdi_prob, r_hat=True
            ),
        ),
        Path(kwargs["figure_dir"]) / "forest.png",
        mlflow_log_artifact=True,
    )

    # DAG を出力
    graph = pm.model_to_graphviz(model)
    dag_filepath = Path(kwargs["figure_dir"]) / "dag"
    graph.render(filename=dag_filepath, format="png", cleanup=True)
    mlflow.log_artifact(f"{dag_filepath}.png")


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("theta_filepath", type=click.Path(exists=True))
@click.argument("csv_output_dir", type=click.Path())
@click.option(
    "--figure_dir", type=click.Path(), default="reports/figures/cvr/"
)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs: Any) -> None:
    """
    メイン処理
    サンプリング結果を分析する
    """
    # init log
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("analysis")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli options
    logger.info(f"args: \n{kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load model, trace, theta
    model, trace = cloudpickle.load(open(kwargs["model_filepath"], "rb"))
    p_a_true, p_b_true, n_a, n_b, observations_a, observations_b = load_theta(
        kwargs["theta_filepath"]
    )

    # 指標を計算
    metrics = {}
    metrics["obs_a"] = calc_summary_of_obs_and_true(observations_a, p_a_true)
    metrics["obs_b"] = calc_summary_of_obs_and_true(observations_b, p_b_true)

    # 確信区間を計算
    hdi_prob = 0.95
    ci = calc_ci(trace.posterior, hdi_prob=hdi_prob)
    metrics.update(ci)
    logger.info(f"metrics: {metrics}")

    # 意思決定に利用する確率などの計算
    prob_summary_df = calc_prob_for_dicision(
        trace, model, p_a_true, p_b_true, observations_a, observations_b
    )

    # 結果を出力
    output_results(
        p_a_true,
        p_b_true,
        model,
        trace,
        metrics,
        hdi_prob,
        prob_summary_df,
        kwargs,
    )

    # cleanup
    mlflow.end_run()
    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
