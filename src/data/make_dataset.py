import logging
import pickle
from typing import Any, Dict

import click
import mlflow
import pandas as pd


def make_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    入力データフレームから観測データを集計し、統計量を計算します。

    Parameters
    ----------
    df : pd.DataFrame
        コンバージョン有無のフラグデータ。
        'obs_a' と 'obs_b' という名前の列を持つことを期待します。
        欠損値 (null) は計算前に除外されます。

    Returns
    -------
    Dict
        各群の試行回数(n)、観測されたコンバージョン率(p_obs)、
        および観測値の配列(obs)を含む辞書。
    """
    results: dict[str, Any] = {}
    for n_key, p_key, obs_key in [
        ("n_a", "p_a_obs", "obs_a"),
        ("n_b", "p_b_obs", "obs_b"),
    ]:
        if obs_key not in df.columns:
            raise ValueError(f"入力データに '{obs_key}' カラムが見つかりません。")

        observations = df[obs_key].dropna()
        results[n_key] = len(observations)
        results[p_key] = observations.mean()
        results[obs_key] = observations.values

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
