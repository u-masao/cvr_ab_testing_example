import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import cloudpickle
import mlflow
import pymc as pm

from src.models.base import define_model


def aggregate(dataset: Dict) -> Tuple[List[int], List[int]]:
    """
    観測データ（0と1の配列）から、試行回数と成功回数を集計します。

    Parameters
    ----------
    dataset : Dict
        'obs_a' と 'obs_b' のキーに観測値の配列を持つ辞書。

    Returns
    -------
    Tuple[List[int], List[int]]
        タプル。(試行回数のリスト, 成功回数のリスト)
    """
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


def sampling(model: pm.Model, kwargs: Any) -> Tuple:
    """MCMCサンプリングを実行します。"""

    with model:
        # pm.sampleのstep引数を指定しない場合、PyMCは最適なサンプラーを自動で選択します。
        # 通常、効率的なNUTS (No-U-Turn Sampler) が使用されます。
        # このサンプルコードでは、以前はMetropolis-Hastingsが指定されていましたが、
        # より現代的で推奨されるアプローチとして、自動選択に任せます。
        trace = pm.sample(
            draws=kwargs["n_sampling"],
            tune=kwargs["n_tune"],
            chains=kwargs["n_chains"],
            random_seed=kwargs["random_seed"],
            progressbar=True,
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
    """
    データセットを読み込み、ベイズモデルを定義し、MCMCサンプリングを実行して、
    結果のモデルとトレースを保存するメイン処理。
    dvcのパイプラインから実行されることを想定しています。
    """
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
