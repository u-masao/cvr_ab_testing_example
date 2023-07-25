import logging
import pickle
from typing import Any, Dict

import click
import mlflow
import pandas as pd


def make_dataset(df: pd.DataFrame) -> Dict:
    """
    入力データを集計し、頻度を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        コンバージョン有無のフラグデータ
        n 行、2 列
        1列目をA群、2列目をB群とする
        null はカウントしない
    """
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
    """
    コンバージョン有無のデータを読み込み頻度を計算する。

    入力データ形式: テキストファイル
    入力フォーマット: CSV、UTF-8(ASCII)、ヘッダあり、2列

    出力データ形式: python pickle, Dict

    引数 INPUT_FILEPATH : str
        入力データのファイルパス

    引数 OUTPUT_FILEPATH : str
        出力データのファイルパス
    """

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
    dataset = make_dataset(raw_df)

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
