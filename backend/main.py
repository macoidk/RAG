from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.model import TaxCodeAssistant

app = FastAPI()
assistant = TaxCodeAssistant(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", persist_directory="/app/db"
)
# mistralai/Mistral-7B-Instruct-v0.3
# mistralai/Mixtral-8x7B-Instruct-v0.1
# mistralai/Mistral-Small-24B-Instruct-2501
# Xwin-LM/Xwin-LM-70B-V0.1


class Query(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class Response(BaseModel):
    answer: str
    sources: list
    metadata: Optional[Dict[str, Any]] = None


@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        response = assistant.process_query(query.text)

        return {
            "answer": response,
            "sources": [],
            "metadata": query.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
