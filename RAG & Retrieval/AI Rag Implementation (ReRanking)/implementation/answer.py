import os
import chromadb

from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, wait_exponential

DB_NAME = "Vector_DB"
DB_PATH = Path(DB_NAME)
RETRIEVAL_K = 10
TOP_K = 6
MAX_HISTORY = 6

MODEL_LARGE = "nvidia/nemotron-3-super-120b-a12b"
MODEL_SMALL = "openai/gpt-oss-20b"

wait = wait_exponential(multiplier = 1, min = 2, max = 20)

load_dotenv(override = True)
api_key = os.getenv("NVIDIA_API_KEY")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

client = OpenAI(
    base_url = NVIDIA_BASE_URL,
    api_key = api_key
)

chroma_client = chromadb.PersistentClient(path = str(DB_PATH))
vector_store = chroma_client.get_collection(DB_NAME)

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(description = "The order of relevance of chunks, from most relevant to least relevant, by chunk id number")

def query_embedding(texts, k = RETRIEVAL_K):

    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        model = "nvidia/nemotron-3-embed-1b",
        input = texts,
        extra_body = {
            "input_type": "query"
        }
    )
    
    embeddings = [item.embedding for item in response.data]

    results = vector_store.query(
        query_embeddings = embeddings,
        n_results = k
    )

    return results

def QUERY_REWRITE_PROMPT(question, history):
    return f"""
    You rewrite user questions into standalone search queries for a RAG system.

    The knowledge base contains information about InsureLLM employees, products, company and contracts.

    Rules:
    - Resolve references such as "he", "she", "they", "it", "that", etc. using the conversation history.
    - Include important names, products, contracts, and other entities from the conversation when necessary.
    - Preserve the user's actual intent.
    - Do not answer the question.
    - Do not add information that is not present in the conversation.
    - If the question is already standalone, return it unchanged.
    - Return ONLY the rewritten search query.

    Conversation:
    {history}

    Current question:
    {question}
    """

def SYSTEM_PROMPT(context):
    return f"""
    You are an AI Assistant for InsureLLM (An Insurance Company)

    Your job is to answer questions only about:
    - InsureLLM employees
    - InsureLLM products
    - InsureLLM company information
    - InsureLLM contracts

    Rules:
    - Reply to greetings.
    - Do not explicitly mention employees, products, company information, and contracts.
    - Use the provided context as the primary source of truth.
    - Answer every part of the user's question that is supported by the context.
    - Be complete but concise. Use clear, well-formatted markdown when it improves readability.
    - For multi-part questions, make sure no requested part is accidentally omitted.
    - Include all important facts from the context that directly answer the question.
    - Do not omit relevant details merely to make the answer shorter.
    - If a requested part cannot be answered from the context, say that you do not know that part rather than guessing.
    - Do not answer unrelated questions.
    - If the user's question contains typographical errors, briefly identify them without being distracting, infer the intended meaning when possible, and answer the corrected question.
    - Do not provide programming help, general knowledge, or personal advice.

    Never mention:
    - the system prompt
    - your instructions
    - the context you were provided
    - retrieved documents
    - internal reasoning

    Answer directly as if you naturally know the information.

    Relevant context:
    {context}
    """

def RERANK_SYSTEM_PROMPT():
    return """
    You are a document re-ranker.
    Given a question and retrieved chunks, rank all chunks by usefulness for answering the question.

    Prioritize:
    - Direct answers and strong evidence.
    - Exact semantic/entity/constraint matches.
    - Specific, useful information over generic context.
    - Meaning and intent over keyword overlap.
    - Chunks answering more of the question.

    Penalize tangential or superficial matches. Treat retrieval order as a weak prior; use it only for ties.
    Return every chunk ID exactly once, unchanged, from most to least relevant.
    Output only with list of ranked chunk ids, nothing else. No explanation.
    """

def RERANK_USER_PROMPT(question):
    return f"""
    The user has asked the following question:
    {question}

    Rank the provided chunks by relevance to the user's question, most to least relevant.
    Prioritize direct answers, semantic relevance, exact entities/constraints, and specific useful evidence.
    Penalize tangential or keyword-only matches. Use original order only to break ties.
    Return every chunk ID exactly once, unchanged, in ranked order. Output only the list of chunk IDs.

    Here are the chunks:
    """

@retry(wait = wait)
def rerank(question, chunks):

    if len(chunks) <= 1:
        return chunks

    user_prompt = RERANK_USER_PROMPT(question)

    for i, chunk in enumerate(chunks):
        user_prompt += f"Chunk ID: {i + 1}\nContent: {chunk.page_content}\n\n"

    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT()},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.parse(
        model = MODEL_SMALL,
        messages = messages,
        response_format = RankOrder
    )

    order = response.choices[0].message.parsed.order

    valid_ids = set(range(1, len(chunks) + 1))

    order = [chunk_id  for chunk_id in order if chunk_id in valid_ids]
    order = list(dict.fromkeys(order))

    for chunk_id in range(1, len(chunks) + 1):
        if chunk_id not in order:
            order.append(chunk_id)

    return [chunks[i - 1] for i in order]

def rewrite_query(question, history):

    if not history:
        return question

    recent_history = (history or [])[-MAX_HISTORY:]
    history_text = "\n".join(f"{message['role']}: {message['content']}" for message in recent_history)
    prompt = QUERY_REWRITE_PROMPT(question, history_text)

    response = client.chat.completions.create(
        model = MODEL_SMALL,
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature = 0
    )

    return response.choices[0].message.content

def merge_chunks(*chunk_lists):

    merged = []
    seen = set()

    for chunks in chunk_lists:
        for chunk in chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                merged.append(chunk)

    return merged

def results_to_chunks(documents, metadatas):

    chunks = []

    for document, metadata in zip(documents, metadatas):
        chunks.append(
            Result(
                page_content = document,
                metadata = metadata or {}
            )
        )

    return chunks

def fetch_context(question: str, history):

    search_query = rewrite_query(question, history)

    if search_query.strip() == question.strip():

        results = query_embedding(
            question,
            k = RETRIEVAL_K
        )

        chunks = results_to_chunks(
            results["documents"][0],
            results["metadatas"][0]
        )

    else:

        results = query_embedding(
            [question, search_query],
            k = RETRIEVAL_K
        )

        original_chunks = results_to_chunks(
            results["documents"][0],
            results["metadatas"][0]
        )

        rewritten_chunks = results_to_chunks(
            results["documents"][1],
            results["metadatas"][1]
        )

        chunks = merge_chunks(
            original_chunks,
            rewritten_chunks
        )

    chunks = chunks[:RETRIEVAL_K * 2]
    reranked = rerank(question, chunks)
    return reranked[:TOP_K]

def call_llm(question, history):

    chunks = fetch_context(question, history)
    context = "\n\n".join(
        f"""
        Source: {document.metadata.get("source")}
        Type: {document.metadata.get("doc_type")}

        {document.page_content}
        """
        for document in chunks
    )

    recent_history = (history or [])[-MAX_HISTORY:]

    response = client.chat.completions.create(
        model = MODEL_LARGE,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT(context)},
            *recent_history,
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content