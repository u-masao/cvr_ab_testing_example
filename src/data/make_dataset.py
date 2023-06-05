import json
import logging
from typing import Any, Dict

import click


def make_dataset(kwargs: Dict) -> Dict:
    return {}


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
