import logging
import pickle
from typing import Any, Dict, Tuple

import click


def sampling(dataset: Dict, kwargs: Dict) -> Tuple:
    trace, model = None, None
    return trace, model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(**kwargs: Any) -> None:
    """メイン処理"""
    logger = logging.getLogger(__name__)
    logger.info("start process")
    logger.info(f"args: {kwargs}")

    with open(kwargs["input_filepath"], "rb") as fo:
        dataset = pickle.load(fo)

    trace, model = sampling(dataset, kwargs)

    logger.info(f"pickle output filepath: {kwargs['output_filepath']}")
    with open(kwargs["output_filepath"], "wb") as fo:
        pickle.dump((trace, model), fo)

    logger.info("complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
