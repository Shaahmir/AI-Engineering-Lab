import os
import chromadb

from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

from llama_index.core import Document
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser as SemanticSplitter

load_dotenv(override = True)
api_key = os.getenv("NVIDIA_API_KEY")

DB_NAME = "Vector_DB"
DB_PATH = Path(DB_NAME)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

client = OpenAI(
    base_url = NVIDIA_BASE_URL,
    api_key = api_key
)

files = list(Path("knowledge-base").rglob("*.md"))
print(f"Found {len(files)} files in the knowledge base")

documents = []

for file in files:
    with open(file, "r", encoding = "utf-8") as f:
        documents.append(
            Document(
                text = f.read(),
                metadata = {
                    "source": str(file),
                    "doc_type": file.parent.name
                }
            )
        )

print(f"Loaded {len(documents)} documents")

embeddings = NVIDIAEmbedding(
    model = "nvidia/nemotron-3-embed-1b",
    api_key = api_key
)

splitter = SemanticSplitter(
    embed_model = embeddings,
    buffer_size = 1,
    breakpoint_percentile_threshold = 80,
)

nodes = splitter.get_nodes_from_documents(documents)
chunks = [{"content": node.get_content(), "metadata": node.metadata} for node in nodes]

def batch_embedding(texts, batch_size):

    embeddings = []

    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model = "nvidia/nemotron-3-embed-1b",
            input = texts[i : i + batch_size],
            extra_body = {
                "input_type": "passage"
            }
        )
        
        embeddings.extend(item.embedding for item in response.data)

    return embeddings

texts = [chunk["content"] for chunk in chunks]
embeddings = batch_embedding(texts, 128)

print(f"Generated embedding {len(embeddings[0])} Dimensions.")

chroma_client = chromadb.PersistentClient(path = str(DB_PATH))

try:
    chroma_client.delete_collection(DB_NAME)
except Exception:
    pass

vector_store = chroma_client.get_or_create_collection(DB_NAME)

ids = [f"Chunk_{i}" for i in range(len(chunks))]
documents = [chunk["content"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

vector_store.add(
    ids = ids,
    documents = documents,
    metadatas = metadatas,
    embeddings = embeddings
)

print(f"Vector database now contains {vector_store.count()} chunks.")