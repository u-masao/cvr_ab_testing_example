import os
from pathlib import Path
from typing import Union

import arviz as az
import matplotlib as mpl
import mlflow
import pymc as pm


def plot_trace(trace: az.InferenceData, model: pm.Model) -> mpl.figure.Figure:
    # save trace plot
    with model:
        axes = pm.plot_trace(trace, compact=False, combined=False)
    fig = make_fig_from_axes(axes)
    fig.suptitle("trace plot")
    return fig


def make_fig_from_axes(axes) -> mpl.figure.Figure:
    for ax in axes.flatten():
        ax.grid()
    fig = axes.ravel()[0].figure
    fig.tight_layout()
    return fig


def savefig(
    fig: mpl.figure.Figure,
    save_path_string: Union[str, Path],
    mlflow_log_artifact: bool = False,
) -> None:
    """
    figure を保存する
    """
    save_path = Path(save_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)
    if mlflow_log_artifact:
        mlflow.log_artifact(save_path)
