import os
import gradio as gr
import pandas as pd

from collections import defaultdict
from dotenv import load_dotenv
from evaluation.evaluator import evaluate_retreivals, evaluate_answers

load_dotenv(override=True)
api_key = os.getenv("NVIDIA_API_KEY")

MRR_GREEN = 0.9
MRR_AMBER = 0.75

NDCG_GREEN = 0.9
NDCG_AMBER = 0.75

COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value: float, metric_type: str) -> str:

    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"

    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:

    color = get_color(value, metric_type)

    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"

    return f"""
    <div style="
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        border-left: 5px solid {color};
        background-color: #f8f9fa;
        color: #111827;
    ">
        <div style="
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #111827;
        ">
            {label}
        </div>
        <div style="
            font-size: 28px;
            font-weight: bold;
            color: {color};
        ">
            {value_str}
        </div>
    </div>
    """


def run_retrieval_evaluation(progress = gr.Progress()):

    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0

    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_retreivals():
        count += 1

        total_mrr += result.MRR
        total_ndcg += result.NDCG
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.MRR)

        progress(prog_value, desc = f"Evaluating test {count}...")

    if count == 0:
        empty_html = """
        <div style="
            padding: 20px;
            text-align: center;
            color: #999;
        ">
            No test questions found.
        </div>
        """

        return empty_html, pd.DataFrame(columns = ["Category", "Average MRR"])

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""
    <div style="padding: 0;">

        {format_metric_html(
            "Mean Reciprocal Rank (MRR)",
            avg_mrr,
            "mrr"
        )}

        {format_metric_html(
            "Normalized DCG (nDCG)",
            avg_ndcg,
            "ndcg"
        )}

        {format_metric_html(
            "Keyword Coverage",
            avg_coverage,
            "coverage",
            is_percentage=True
        )}

        <div style="
            margin-top: 20px;
            padding: 10px;
            background-color: #d4edda;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #c3e6cb;
        ">
            <span style="
                font-size: 14px;
                color: #155724;
                font-weight: bold;
            ">
                ✓ Evaluation Complete: {count} tests
            </span>
        </div>

    </div>
    """

    category_data = []

    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)

        category_data.append({
            "Category": category,
            "Average MRR": avg_cat_mrr
        })

    df = pd.DataFrame(category_data)

    return final_html, df


def run_answer_evaluation(progress = gr.Progress()):

    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0

    category_accuracy = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_answers():
        count += 1

        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        category_accuracy[test.category].append(result.accuracy)

        progress(prog_value, desc = f"Evaluating test {count}...")

    if count == 0:
        empty_html = """
        <div style="
            padding: 20px;
            text-align: center;
            color: #999;
        ">
            No test questions found.
        </div>
        """

        return empty_html, pd.DataFrame(columns = ["Category", "Average Accuracy"])

    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    final_html = f"""
    <div style="padding: 0;">

        {format_metric_html(
            "Accuracy",
            avg_accuracy,
            "accuracy",
            score_format = True
        )}

        {format_metric_html(
            "Completeness",
            avg_completeness,
            "completeness",
            score_format = True
        )}

        {format_metric_html(
            "Relevance",
            avg_relevance,
            "relevance",
            score_format = True
        )}

        <div style="
            margin-top: 20px;
            padding: 10px;
            background-color: #d4edda;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #c3e6cb;
        ">
            <span style="
                font-size: 14px;
                color: #155724;
                font-weight: bold;
            ">
                ✓ Evaluation Complete: {count} tests
            </span>
        </div>

    </div>
    """

    category_data = []

    for category, accuracy_scores in category_accuracy.items():
        avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        category_data.append({
            "Category": category,
            "Average Accuracy": avg_cat_accuracy
        })

    df = pd.DataFrame(category_data)

    return final_html, df


def main():

    theme = gr.themes.Soft(font = ["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title = "RAG Evaluation Dashboard") as app:

        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown(
            "Evaluate retrieval and answer quality "
            "for the Insurellm RAG system"
        )

        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_button = gr.Button(
            "Run Evaluation",
            variant = "primary",
            size = "lg"
        )

        with gr.Row():

            with gr.Column(scale = 1):
                retrieval_metrics = gr.HTML(
                    """
                    <div style="
                        padding: 20px;
                        text-align: center;
                        color: #999;
                    ">
                        Click 'Run Evaluation' to start
                    </div>
                    """
                )

            with gr.Column(scale = 1):
                retrieval_chart = gr.BarPlot(
                    x = "Category",
                    y = "Average MRR",
                    title = "Average MRR by Category",
                    y_lim = [0, 1],
                    height = 400
                )

        gr.Markdown("## 💬 Answer Evaluation")

        answer_button = gr.Button(
            "Run Evaluation",
            variant = "primary",
            size = "lg"
        )

        with gr.Row():

            with gr.Column(scale = 1):
                answer_metrics = gr.HTML(
                    """
                    <div style="
                        padding: 20px;
                        text-align: center;
                        color: #999;
                    ">
                        Click 'Run Evaluation' to start
                    </div>
                    """
                )

            with gr.Column(scale = 1):
                answer_chart = gr.BarPlot(
                    x = "Category",
                    y = "Average Accuracy",
                    title = "Average Accuracy by Category",
                    y_lim = [1, 5],
                    height = 400,
                )

        retrieval_button.click(
            fn = run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart]
        )

        answer_button.click(
            fn = run_answer_evaluation,
            outputs = [answer_metrics, answer_chart]
        )

    app.launch(
        inbrowser = True,
        theme = theme,
    )

if __name__ == "__main__":
    main()