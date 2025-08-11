"""
共通のモデル定義を格納するモジュール。
"""

from typing import List

import pymc as pm


def define_model(trials: List[int], successes: List[int]) -> pm.Model:
    """
    A/Bテスト分析のためのPyMCモデルを定義します。

    このモデルは、2つのグループ（AとB）のコンバージョン率を、
    独立したベルヌーイ分布に従うと仮定します。
    各グループのコンバージョン率の事前分布として、無情報なベータ分布を使用します。

    Parameters
    ----------
    trials : List[int]
        各グループの試行回数のリスト。例: [n_a, n_b]
    successes : List[int]
        各グループの成功（コンバージョン）数のリスト。例: [conversions_a, conversions_b]

    Returns
    -------
    pm.Model
        定義されたPyMCモデル。
    """
    if len(trials) != 2 or len(successes) != 2:
        raise ValueError("trialsとsuccessesは、2つの要素（グループAとB）を持つリストでなければなりません。")

    with pm.Model() as model:
        # 事前分布: 各グループのコンバージョン率pに対する無情報事前分布。
        # Beta(1, 1)は、0から1までの一様分布に相当します。
        p = pm.Beta("p", alpha=1.0, beta=1.0, shape=2)

        # 尤度: 観測されたデータ（成功数）を二項分布でモデル化します。
        _ = pm.Binomial("y", n=trials, p=p, observed=successes, shape=2)

        # 決定論的変数: 分析に有用な派生的な量を定義します。
        # uplift: Bのp - Aのp
        _ = pm.Deterministic("uplift", p[1] - p[0])
        # relative_uplift: (Bのp - Aのp) / Aのp
        _ = pm.Deterministic("relative_uplift", p[1] / p[0] - 1.0)

    return model
