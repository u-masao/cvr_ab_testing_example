import json
import logging
from typing import Any, Dict

import click
from scipy.status import bernoulli


def make_dataset(kwargs: Dict) -> Dict:
    results = {}
    results["n_a"] = kwargs["n_a"]
    results["n_b"] = kwargs["n_b"]
    results["p_a"] = kwargs["p_a"]
    results["p_b"] = kwargs["p_b"]
    results["obs_a"] = bernoulli.rvs(p=kwargs["p_a"], size=kwargs["n_a"])
    results["obs_b"] = bernoulli.rvs(p=kwargs["p_b"], size=kwargs["n_b"])
    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(**kwargs: Any) -> None:
    """メイン処理"""
    logger = logging.getLogger(__name__)
    logger.info("start process")

    dataset = make_dataset(kwargs)

    logger.info(f"json output filepath: {kwargs['output_filepath']}")
    with open(kwargs["output_filepath"], "w") as fo:
        json.dump(dataset, fo, indent=4)

    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
