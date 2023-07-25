import logging
import pickle
from typing import Any, Dict

import click
import mlflow
from scipy.stats import bernoulli


def make_dataset(kwargs: Dict) -> Dict:
    """パラメータに基づきベルヌーイ分布の観測値を生成する"""

    results = {}
    for n, p, obs in [
        ["n_a", "p_a_true", "obs_a"],
        ["n_b", "p_b_true", "obs_b"],
    ]:
        # save parameters
        results[n] = kwargs[n]
        results[p] = kwargs[p]

        # sampling
        results[obs] = bernoulli.rvs(
            p=kwargs[p], size=kwargs[n], random_state=kwargs["random_state"]
        )

    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
@click.option("--n_a", type=int, default=1500)
@click.option("--n_b", type=int, default=750)
@click.option("--p_a_true", type=float, default=0.04)
@click.option("--p_b_true", type=float, default=0.05)
@click.option("--random_state", type=int, default=1234)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs: Any) -> None:
    """
    パラメーターに基づいてベルヌーイ分布の観測値を生成し集計する。
    """

    # init logger
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("make_observed")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli options
    logger.info(f"args: \n{kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # make observed
    dataset = make_dataset(kwargs)

    # output
    logger.info(f"pickle output filepath: {kwargs['output_filepath']}")
    pickle.dump(dataset, open(kwargs["output_filepath"], "wb"))

    # cleanup
    logger.info(f"data.keys(): {dataset.keys()}")
    mlflow.end_run()
    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
