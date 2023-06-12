import logging
import pickle
from typing import Any, Dict

import click
import mlflow
import pandas as pd


def make_dataset(df: pd.DataFrame, kwargs: Dict) -> Dict:
    results = {}
    for index, (n, p, obs) in enumerate(
        [
            ["n_a", "p_a_true", "obs_a"],
            ["n_b", "p_b_true", "obs_b"],
        ]
    ):
        results[n] = len(df.iloc[:, index].dropna())
        results[p] = df.iloc[:, index].mean()
        results[obs] = df.iloc[:, index].values

    return results


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs: Any) -> None:
    """メイン処理"""

    # init logger
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("make_dataset")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli options
    logger.info(f"args: \n{kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load data
    raw_df = pd.read_csv(kwargs["input_filepath"])

    # make observed
    dataset = make_dataset(raw_df, kwargs)

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
