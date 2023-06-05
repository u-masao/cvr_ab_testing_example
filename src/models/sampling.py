import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import pymc as pm


def sampling(dataset: Dict, kwargs: Dict) -> Tuple:

    # init logger
    logger = logging.getLogger(__name__)

    # count obs
    trials = []
    successes = []
    for key in ["obs_a", "obs_b"]:
        trials.append(len(dataset[key]))
        successes.append(dataset[key].sum())
    logger.info(f"trials: {trials}")
    logger.info(f"successes: {successes}")

    # init model
    model = pm.Model()

    # define model
    with model:
        p = pm.Beta("p", alpha=1.0, beta=1.0, shape=2)
        obs = pm.Binomial(  # noqa: F841
            "y", n=trials, p=p, shape=2, observed=successes
        )
        relative_uplift = pm.Deterministic(  # noqa: F841
            "relative_uplift", p[1] / p[0] - 1.0
        )

    # sampling
    with model:
        trace = pm.sample(kwargs["n_sampling"])

    return trace, model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--n_sampling", type=int, default=10000)
def main(**kwargs: Any) -> None:
    """メイン処理"""
    logger = logging.getLogger(__name__)
    logger.info("start process")
    logger.info(f"args: {kwargs}")

    with open(kwargs["input_filepath"], "rb") as fo:
        dataset = pickle.load(fo)

    trace, model = sampling(dataset, kwargs)

    logger.info(f"pickle output filepath: {kwargs['output_filepath']}")
    output_filepath = Path(kwargs["output_filepath"])
    os.makedirs(output_filepath.parent, exist_ok=True)
    with open(output_filepath, "wb") as fo:
        pickle.dump((trace, model), fo)

    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
