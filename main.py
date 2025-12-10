
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from memory_extractor import extract_personality
from memory_store import get_memory, update_memory, merge_memory
from personality_engine import transform_personality, PERSONALITY_STYLES

app = FastAPI(
    title="AI Memory + Personality Engine",
    version="1.0.0"
)

def health_check():
    return {"status": "ok"}

class MemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    messages: List[str] = Field(..., min_items=1)


@app.post("/extract-memory")
async def extract_memory(req: MemoryRequest):
    try:
        # 1. Extract memory using LLM
        llm_result = extract_personality(req.messages)

        new_memory = {
            "preferences": llm_result.preferences,
            "facts": llm_result.facts,
            "emotions": llm_result.emotions,
        }

        # 2. Load existing memory
        existing = get_memory(req.user_id)

        # 3. Merge
        merged = merge_memory(existing, new_memory)
        merged["user_id"] = req.user_id

        # 4. Save
        update_memory(req.user_id, merged)

        return {"status": "success", "memory": merged}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class TransformRequest(BaseModel):
    base_reply: str = Field(..., min_length=1)
    personality: str = Field(..., min_length=1)


@app.post("/transform-reply")
async def transform_reply(req: TransformRequest):
    if req.personality not in PERSONALITY_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid personality. Choose from: {', '.join(PERSONALITY_STYLES.keys())}"
        )

    try:
        transformed = transform_personality(req.base_reply, req.personality)

        return {
            "before": req.base_reply,
            "after": transformed,
            "personality": req.personality
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
