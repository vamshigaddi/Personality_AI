import json
from google import genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class ExtractedMemory(BaseModel):
    preferences: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

def extract_personality(messages: List[str]) -> ExtractedMemory:
    """
    Sends the user's last 30 messages to Gemini and extracts:
    - preferences
    - emotional patterns
    - long-term facts
    """

    text_block = "\n".join(messages)

    full_prompt = f"""
    You are a Memory Extraction Engine.

    Task:
    Analyze the user's last 30 messages and extract ONLY stable long-term memories:
    
    - preferences (food, hobbies, interests, tools they like)
    - facts (job, long-term goals, skills, repeating habits)
    - emotional patterns (stressed often, optimistic, sarcastic, anxious, etc.)

    Extraction constraints:
    - Only include information supported by MULTIPLE messages.
    - Never guess or infer from a single message.
    - Ignore temporary moods or single-message emotions.
    - Output ONLY valid JSON matching this schema:

    {{
        "preferences": ["string"],
        "facts": ["string"],
        "emotions": ["string"]
    }}

    Now analyze these user messages:

    -------------------
    {text_block}
    -------------------
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[full_prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractedMemory,
        }
    )

    # Safely parse the output
    try:
        data = json.loads(response.text)
        return ExtractedMemory(**data)
    except (json.JSONDecodeError, ValidationError):
        return ExtractedMemory()
