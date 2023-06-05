import logging
import pickle
from typing import Any, Dict

import click
from scipy.stats import bernoulli


def make_dataset(kwargs: Dict) -> Dict:
    results = {}
    for n, p, obs in [["n_a", "p_a", "obs_a"], ["n_b", "p_b", "obs_b"]]:
        results[n] = kwargs[n]
        results[p] = kwargs[p]
        results[obs] = bernoulli.rvs(
            p=kwargs[p], size=kwargs[n], random_state=kwargs["random_state"]
        )
    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
@click.option("--random_state", type=int, default=1234)
@click.option("--n_a", type=int, default=1500)
@click.option("--n_b", type=int, default=750)
@click.option("--p_a", type=float, default=0.04)
@click.option("--p_b", type=float, default=0.05)
def main(**kwargs: Any) -> None:
    """メイン処理"""
    logger = logging.getLogger(__name__)
    logger.info("start process")

    dataset = make_dataset(kwargs)

    logger.info(f"pickle output filepath: {kwargs['output_filepath']}")
    with open(kwargs["output_filepath"], "wb") as fo:
        pickle.dump(dataset, fo)

    logger.info(f"data: {dataset}")
    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
