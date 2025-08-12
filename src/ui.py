import gradio as gr

from src.analysis import (
    run_bayesian_analysis,
    run_chisquared_test,
    run_power_analysis,
)
from src.ui_texts import (
    text_about_baysian_analysis,
    text_about_chi_squared_test,
    text_about_power_analysis,
    text_about_this_tool,
    text_bayesien_vs_chi_squared,
)


def create_header():
    """Gradio UIのヘッダー部分を作成します"""
    gr.Markdown("# A/Bテスト分析ツール (ベイズ推定 vs カイ二乗検定)")
    with gr.Accordion("❓ このツールで何ができるか", open=False):
        gr.Markdown(text_about_this_tool)


def create_main_components(demo):
    # --- 入力コンポーネント ---
    gr.Markdown("## 1. テスト結果の入力")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### A群")
            n_a = gr.Number(1000, label="試行回数 (サンプルサイズ)")
            conversion_a = gr.Number(53, label="コンバージョン数")
        with gr.Column():
            gr.Markdown("### B群")
            n_b = gr.Number(100, label="試行回数 (サンプルサイズ)")
            conversion_b = gr.Number(10, label="コンバージョン数")

    with gr.Accordion("⚙️  ベイズ分析の詳細設定", open=False):
        n_sampling = gr.Number(4000, label="サンプリング回数 (draws)")
        n_tune = gr.Number(1000, label="チューニング回数 (tune)")
        n_chains = gr.Number(4, label="チェーン数")
        hdi_prob = gr.Number(
            0.95, label="Highest Density Interval (HDI:最高事後密度区間)"
        )
        random_seed = gr.Number(1234, label="ランダムシード")

    create_power_analysis_components(demo)
    start_button = gr.Button("分析開始", variant="primary")

    # --- 出力コンポーネント ---
    gr.Markdown("## 2. 分析結果")
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ベイズ分析の結果")
            with gr.Tabs():
                with gr.TabItem("📈 分布"):
                    distribution = gr.Plot()
                with gr.TabItem("📊 サマリー"):
                    params = gr.Plot()
                with gr.TabItem("🕸️ モデル"):
                    model_img = gr.Image()
                with gr.TabItem("🛰️ トレースプロット"):
                    trace_plot = gr.Plot()
        with gr.Column(scale=1):
            gr.Markdown("### カイ二乗検定の結果")
            chisq_result_df = gr.DataFrame(
                headers=["指標", "値"], datatype=["str", "str"], label="検定結果"
            )

    start_button.click(
        fn=run_chisquared_test,
        inputs=[
            n_a,
            conversion_a,
            n_b,
            conversion_b,
            hdi_prob,
        ],
        outputs=[
            chisq_result_df,
        ],
    ).then(
        fn=run_bayesian_analysis,
        inputs=[
            n_a,
            conversion_a,
            n_b,
            conversion_b,
            n_sampling,
            n_tune,
            n_chains,
            random_seed,
            hdi_prob,
        ],
        outputs=[
            distribution,
            model_img,
            trace_plot,
            params,
        ],
    )


def create_power_analysis_components(demo):
    """サンプルサイズ計算コンポーネントを作成します"""
    with gr.Accordion("🙏 サンプルサイズ計算", open=False):
        with gr.Row():
            with gr.Column():
                power_analysis_inputs = {
                    "baseline_cvr": gr.Number(
                        0.053,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        label="ベースラインCVR",
                    ),
                    "lift": gr.Number(
                        0.01,
                        minimum=-1.0,
                        maximum=1.0,
                        step=0.001,
                        label="検出したいCVRの増加量",
                    ),
                    "alpha": gr.Number(
                        0.05,
                        minimum=0.01,
                        maximum=0.5,
                        step=0.01,
                        label="有意水準(Alpha)",
                    ),
                    "power": gr.Number(
                        0.80,
                        minimum=0.5,
                        maximum=0.99,
                        step=0.01,
                        label="検出力(1-Beta)",
                    ),
                    "ratio": gr.Number(
                        0.10,
                        minimum=0.0,
                        maximum=1000,
                        step=0.01,
                        label="A群に対するB群のサンプルサイズ比",
                    ),
                }
            with gr.Column():
                power_analysis_sample_size_a = gr.Number(label="A 群のサンプルサイズ")
                power_analysis_sample_size_b = gr.Number(label="B 群のサンプルサイズ")
                with gr.Accordion("🔍分析結果の詳細", open=False):
                    gr.Markdown("Cohen's h で効果量を計算し、サンプルサイズを計算しています")
                    power_analysis_output = gr.DataFrame()
    gr.on(
        triggers=[x.change for x in power_analysis_inputs.values()],
        fn=run_power_analysis,
        inputs=[x for x in power_analysis_inputs.values()],
        outputs=[
            power_analysis_sample_size_a,
            power_analysis_sample_size_b,
            power_analysis_output,
        ],
    )


def create_explanation_section():
    """解説セクションを作成します"""
    with gr.Accordion("💡 結果の解釈と解説", open=False):
        with gr.Tabs():
            with gr.TabItem("ベイズ分析の解説"):
                gr.Markdown(text_about_baysian_analysis)
            with gr.TabItem("カイ二乗検定の解説"):
                gr.Markdown(text_about_chi_squared_test)
            with gr.TabItem("検出力分析とは？"):
                gr.Markdown(text_about_power_analysis)
            with gr.TabItem("ベイズ vs カイ二乗"):
                gr.Markdown(text_bayesien_vs_chi_squared)


def build_ui(demo):
    """Gradio UI全体を構築し、イベントリスナーを設定します"""
    create_header()
    create_main_components(demo)
    create_explanation_section()


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    build_ui(demo)


if __name__ == "__main__":
    demo.launch(share=False)
