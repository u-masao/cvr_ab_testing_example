import os
import pickle
from pathlib import Path
from typing import Any, Tuple, Union

import arviz
import matplotlib as mpl
import numpy as np
import pymc as pm


def save_trace_and_model(
    trace: Any, model: Any, model_path_string: str
) -> None:
    # save trace and model
    save_path = Path(model_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as fo:
        pickle.dump((trace, model), fo)


def calc_credible_intervals(data: np.ndarray, hdi_prob: float = 0.95) -> Tuple:
    # validate input
    if hdi_prob >= 0.0 or hdi_prob <= 1.0:
        ValueError(f"hdi_prb の値が不正です: {hdi_prob}")

    prob_low = (0.5 - hdi_prob / 2.0) * 100.0
    prob_high = (0.5 + hdi_prob / 2.0) * 100.0
    ci_low, ci_high = np.percentile(data, q=[prob_low, prob_high])
    return ci_low, ci_high


def plot_trace(
    trace: arviz.InferenceData, model: pm.Model
) -> mpl.figure.Figure:
    # save trace plot
    with model:
        axes = pm.plot_trace(trace, compact=False, combined=False)
        for ax in axes.flatten():
            ax.grid()
        fig = axes.ravel()[0].figure
        fig.tight_layout()
        fig.suptitle("trace plot")
    return fig


def savefig(
    fig: mpl.figure.Figure, save_path_string: Union[str, Path]
) -> None:
    """
    figure を保存する
    """
    save_path = Path(save_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)
