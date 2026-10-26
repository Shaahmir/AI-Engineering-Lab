import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "test_set.jsonl")

class TestQuestion(BaseModel):
    question: str = Field(description = "The question to ask from RAG system")
    keywords: list[str] = Field(description = "Key words that must appear in retreived context")
    reference_answer: str = Field(description = "The reference answer to for this question")
    category: str = Field(description = "Question category (e.g: direct_fact, spanning, temporal)")

def load_testset() -> list[TestQuestion]:
    
    testset = []

    with open(TEST_FILE, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            testset.append(TestQuestion(**data))

    return testset