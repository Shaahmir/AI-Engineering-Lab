import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override = True)
api_key = os.getenv("NVIDIA_API_KEY")

DB_NAME = "Vector_DB"
DB_PATH = Path(DB_NAME)

files = list(Path("knowledge-base").rglob("*.md"))
print(f"Found {len(files)} files in the knowledge base")

documents = []

for file in files:
    with open(file, "r", encoding = "utf-8") as f:
        documents.append(
            Document(
                page_content = f.read(),
                metadata = {
                    "source": str(file),
                    "doc_type": file.parent.name
                }
            )
        )

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

embeddings = OpenAIEmbeddings(
    model = "nvidia/nemotron-3-embed-1b",
    base_url = NVIDIA_BASE_URL,
    api_key = api_key,
    chunk_size = 256,
    check_embedding_ctx_length = False,
    extra_body = {
        "input_type": "passage"
    }
)

if DB_PATH.exists():
    Chroma(
        embedding_function = embeddings,
        persist_directory = str(DB_PATH)
    ).delete_collection()

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = str(DB_PATH),
)

vector_store.add_documents(
    documents = chunks
)

print(f"Created a VectorStore with {vector_store._collection.count()} documents")