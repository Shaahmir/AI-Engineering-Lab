import os
import math
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_testset
from implementation.answer import call_llm, fetch_context

from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential

load_dotenv(override = True)
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key,
)

CONCURRENT_REQUESTS = 12
wait = wait_exponential(multiplier = 1, min = 2, max = 10)

class EvalMetrics(BaseModel):
    MRR: float = Field(description = "Mean Reciprocal Rank")
    NDCG: float = Field(description = "Normalized Discounted Cummulative Gain")
    keywords_found: int = Field(description = "Number of kewords found in Top-K results")
    total_keywords: int = Field(description = "Total number of keywords to find")
    keyword_coverage: float = Field(description = "Percentage of keywords found")

class AnswerEval(BaseModel):

    feedback: str = Field("Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context")
    accuracy: float = Field("How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3.")
    completeness: float = Field("How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included.")
    relevance: float = Field("How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information.")

# MRR Calculating Function

def calculate_MRR(keyword: str, retreived_docs: list):

    for rank, doc in enumerate(retreived_docs, start = 1):

        if keyword.lower() in doc.page_content.lower():
            return 1.0 / rank

    return 0.0

# DCG Calculating Function

def calculate_DCG(relevances: list[int], K: int):

    DCG = 0.0

    for i in range(min(K, len(relevances))):
        DCG += relevances[i] / math.log2(i + 2)

    return DCG

# NDCG Calculating Function

def calculate_NDCG(keyword: str, retreived_docs: list, K = 5):

    relevances = [1 if keyword.lower() in doc.page_content.lower() else 0 for doc in retreived_docs[:K]]
    DCG = calculate_DCG(relevances, K)

    ideal_relevances = sorted(relevances, reverse = True)
    IDCG = calculate_DCG(ideal_relevances, K)

    return DCG / IDCG if IDCG > 0 else 0.0

def retreival_metrics(test_question: TestQuestion, K = 5):

    retreived_docs = fetch_context(test_question.question, None)

    MRR = [calculate_MRR(keyword, retreived_docs) for keyword in test_question.keywords]
    AVG_MRR = sum(MRR) / len(MRR) if MRR else 0.0

    NDCG = [calculate_NDCG(keyword, retreived_docs) for keyword in test_question.keywords]
    AVG_NDCG = sum(NDCG) / len(NDCG) if NDCG else 0.0

    keywords_found = sum(1 for score in MRR if score > 0)
    total_keywords = len(test_question.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return EvalMetrics(
        MRR = AVG_MRR,
        NDCG = AVG_NDCG,
        keywords_found = keywords_found,
        total_keywords = total_keywords,
        keyword_coverage = keyword_coverage
    )
@retry(wait = wait)
def evaluate_answer(test: TestQuestion):

    print(f"[START] {test.question}", flush = True)
    generated_answer = call_llm(test.question, [])
    print(f"[ANSWER DONE] {test.question}", flush = True)

    user_prompt = f"""
        Question:
        {test.question}

        Generated Answer:
        {generated_answer}

        Reference Answer:
        {test.reference_answer}

        Please evaluate the generated answer on three dimensions:
        1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
        2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
        3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

        Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.
    """

    judge_messages = [
        {"role": "system", "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers."},
        {"role": "user", "content": user_prompt}
    ]

    print(f"[JUDGE START] {test.question}", flush = True)

    judge_response = client.chat.completions.parse(
        model = "nvidia/nemotron-3-super-120b-a12b",
        messages = judge_messages,
        response_format = AnswerEval
    )

    print(f"[JUDGE DONE] {test.question}", flush = True)

    answer_eval = judge_response.choices[0].message.parsed

    return answer_eval, generated_answer

def evaluate_retreivals():

    testset = load_testset()
    total_testset = len(testset)

    if total_testset == 0:
        return

    print(f"\nStarting paralel answer evaluation for {total_testset} tests ...", flush = True)
    print(f"Sending {CONCURRENT_REQUESTS} requests", flush = True)

    completed = 0

    with ThreadPoolExecutor(max_workers = CONCURRENT_REQUESTS) as executor:

        futures = {
            executor.submit(
                retreival_metrics,
                question
            ): question
            for question in testset
        }

        for future in as_completed(futures):

            question = futures[future]

            try:
                result = future.result()

            except Exception as e:
                print(f"[ERROR] {question.question}", flush = True)
                print(f"{type(e).__name__}: {e}", flush = True)
                continue

            completed += 1
            progress = completed / total_testset

            print(f"[PROGRESS] {completed}/{total_testset} - {progress * 100:.2f}%", flush = True)

            yield question, result, progress

    # for index, question in enumerate(testset):
    #     result = retreival_metrics(question)
    #     progress = (index + 1) / total_testset
    #     yield question, result, progress

def evaluate_answers():

    testset = load_testset()
    total_testset = len(testset)

    if total_testset == 0:
        return

    print(f"\nStarting paralel answer evaluation for {total_testset} tests ...", flush = True)
    print(f"Sending {CONCURRENT_REQUESTS} requests", flush = True)

    completed = 0

    with ThreadPoolExecutor(max_workers = CONCURRENT_REQUESTS) as executor:

        futures = {
            executor.submit(
                evaluate_answer,
                question
            ): question
            for question in testset
        }

        for future in as_completed(futures):

            question = futures[future]

            try:
                result = future.result()

            except Exception as e:
                print(f"[ERROR] {question.question}", flush = True)
                print(f"{type(e).__name__}: {e}", flush = True)
                continue

            completed += 1
            progress = completed / total_testset

            print(f"[PROGRESS] {completed}/{total_testset} - {progress * 100:.2f}%", flush = True)

            yield question, result[0], progress

    # for index, question in enumerate(testset):
    #     result = evaluate_answer(question)[0]
    #     progress = (index + 1) / total_testset
    #     yield question, result, progress