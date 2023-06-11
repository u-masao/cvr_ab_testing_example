import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import cloudpickle
import mlflow
import pymc as pm


def aggregate(dataset: Dict) -> Tuple:
    """0と1の観測値を試行回数と成功回数に集計"""

    # init logger
    logger = logging.getLogger(__name__)

    # count obs
    trials = []
    successes = []
    for key in ["obs_a", "obs_b"]:
        # 試行回数を計算
        trials.append(len(dataset[key]))

        # 成功回数を計算
        successes.append(dataset[key].sum())

    logger.info(f"trials: {trials}")
    logger.info(f"successes: {successes}")

    return trials, successes


def define_model(trials: List, successes: List) -> pm.Model:
    """モデルを定義"""
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
        uplift = pm.Deterministic("uplift", p[1] - p[0])  # noqa: F841
    return model


def sampling(model: pm.Model, kwargs: Any) -> Tuple:
    """sampling"""

    with model:
        # set Metropolis-Hastings sampling step
        step = pm.Metropolis()

        # sampling
        trace = pm.sample(
            kwargs["n_sampling"],
            tune=kwargs["n_tune"],
            step=step,
            chains=kwargs["n_chains"],
            random_seed=kwargs["random_seed"],
        )

    return model, trace


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--n_sampling", type=int, default=10000)
@click.option("--n_chains", type=int, default=5)
@click.option("--n_tune", type=int, default=2000)
@click.option("--random_seed", type=int, default=1234)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs: Any) -> None:
    """メイン処理"""

    # init log
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("sampling")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli options
    logger.info(f"args: \n{kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load dataset
    dataset = pickle.load(open(kwargs["input_filepath"], "rb"))

    # 観測値を集計
    trials, successes = aggregate(dataset)

    # モデルを定義
    model = define_model(trials, successes)

    # サンプリング
    model, trace = sampling(model, kwargs)

    # save model and trace
    output_filepath = Path(kwargs["output_filepath"])
    os.makedirs(output_filepath.parent, exist_ok=True)
    cloudpickle.dump((model, trace), open(output_filepath, "wb"))

    # cleanup
    mlflow.end_run()
    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
