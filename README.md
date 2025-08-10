cvr_ab_testing_example
======================

このソフトウェアは成功と失敗の 2 値をとる試行について A/B テストを行った結果を分析します。

分析方法
--------

A群とB群を独立したベルヌーイ分布とみなしてパラメーターをベイズ推定します。

- DAG
  - ![DAG](https://github.com/u-masao/cvr_ab_testing_example/blob/main/reports/figures/real/dag.png)

動作確認環境
------------

- OS: Ubuntu 22.04
- Python: 3.10

インストール方法
----------------


poetry のインストール

```
$ pip install poetry
```

ライブラリのインストール

```
$ git clone https://github.com/u-masao/cvr_ab_testing_example.git
$ cd cvr_ab_testing_example
$ poetry install
```

データの作成
------------

実験結果を CSV ファイルで保存します。

- ファイル保存先: data/raw/observed_real.csv
- ファイル形式: CSV, UTF-8, ヘッダあり
- 試行していない箇所は NULL とする
- 行の関連は見ない(対応のない2群の分析)

```:入力データ例
obs_a,obs_b
1,0
1,0
1,1
0,1
,1
,0
,1
```


実行方法
--------

```
$ poetry run dvc repro visualization_real
```

実行結果
--------

- チャート
  - reports/figures/real/

- 数値
  - data/processed/real/


結果の例
--------

![結果の例](https://github.com/u-masao/cvr_ab_testing_example/blob/main/reports/figures/real/distribution.png)


- A 群と B 群のベルヌーイ分布のパラメーターの分布を推定した結果を表示

- 確信区間を読み取れる

- uplift や 相対 uplift の累積分布から「Bが負ける確率」などを読み取れる


Web UI について
---------------

以下のコマンドで Web UI が利用可能です。

```
make ui
```

ブラウザで http://localhost:7860/ にアクセスして下さい。


![docs/Web-ui-screenshot.png]


