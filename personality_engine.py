from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

PERSONALITY_STYLES = {
    "calm_mentor": {
        "description": "A calm, wise, steady mentor with gentle guidance.",
        "rules": [
            "Use warm, steady tone.",
            "Avoid urgency.",
            "Provide wisdom concisely.",
            "No slang, no emojis.",
        ]
    },
    "witty_friend": {
        "description": "A playful, witty best friend who teases lightly.",
        "rules": [
            "Use light sarcasm tastefully.",
            "Use emojis sparingly but naturally.",
            "Keep sentences short, fun, and casual.",
            "Avoid deep analysis.",
        ]
    },
    "therapist": {
        "description": "A deeply empathetic therapist who validates emotions.",
        "rules": [
            "Acknowledge feelings explicitly.",
            "Use calm and validating language.",
            "Do not joke or use emojis.",
            "Avoid giving commands.",
        ]
    },
    "confident_coach": {
        "description": "A high-energy motivational coach.",
        "rules": [
            "Use short punchy sentences.",
            "Sound confident and uplifting.",
            "Use energetic language.",
            "Optional light emojis (🔥💪) but limited.",
        ]
    },
}


def transform_personality(base_reply: str, style: str) -> str:
    if style not in PERSONALITY_STYLES:
        raise ValueError("Invalid style. Choose from: " + ", ".join(PERSONALITY_STYLES.keys()))

    style_data = PERSONALITY_STYLES[style]

    prompt = f"""
You are a Personality Transformation Engine.

Transform the assistant reply into the target persona.

Persona: {style}
Persona Description: {style_data['description']}
Persona Rules:
{chr(10).join(f"- {r}" for r in style_data['rules'])}

Transformation Requirements:
- Keep the same meaning.
- Rewrite the tone, rhythm, and voice to match the persona.
- Do NOT add prefacing such as “Here is your rewritten text”.
- Do NOT add new facts.
- Output ONLY the rewritten message.

Original Message:
<<<
{base_reply}
>>>

Now rewrite it in the target persona.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={
            "response_mime_type": "text/plain",
        }
    )

    return response.text.strip()
