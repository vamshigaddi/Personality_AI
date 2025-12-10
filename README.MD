# 🧠 AI Memory Extraction + Personality Engine  
A modular AI system that:

1. **Analyzes the last 30 user messages** to extract  
   - stable preferences  
   - long-term facts  
   - emotional patterns  
2. **Stores memory in MongoDB** with confidence-based reinforcement  
3. **Transforms assistant replies** into custom personalities  
   (calm mentor, witty friend, therapist, confident coach)

- This project showcases **memory systems**, **reasoning**, **prompt design**, **API architecture**, and **personality-controlled responses** — core skills for Companion-AI systems.

---

## ⭐ Features

### ✅ Memory Extraction (LLM-powered)
- Extracts stable long-term user patterns  
- Avoids single-message noise  
- Ensures JSON correctness via Pydantic  
- Structured reasoning + strong prompt design  

### ✅ Memory Store with Confidence Tracking
- Uses MongoDB  
- Stores memory **only if repeated multiple times**  
- Prevents hallucinated or unstable memory  
- Keeps `history_counts` to track confidence  

### ✅ Personality Engine
Transforms replies into specific personas:
- 🧘 **Calm Mentor**
- 😄 **Witty Friend**
- 🩺 **Therapist**
- 💪 **Confident Coach**

Each personality includes:
- Description  
- Tone rules  
- Expression style  
- Constraints for stable voice consistency  

### ✅ FastAPI Backend
Exposes two major APIs:
- `/extract-memory`
- `/transform-reply`

Clean, modular, production-friendly.

---

## 📂 Project Structure
```
assignment/
│
├── api.py                   # FastAPI server entrypoint
├── memory_extractor.py      # LLM-based memory extraction
├── memory_store.py          # MongoDB persistence + merge logic
├── personality_engine.py    # Tone transformation engine
├── .env                     # API keys (not included)
└── requirements.txt
```

### 🚀 How It Works
### 1. Memory Extraction Pipeline
```
User Messages (30) → extract_personality() → LLM Output (JSON)
             ↓
       merge_memory()
             ↓
     Stored in MongoDB
```

- The extractor retrieves:

 - "preferences"

 - "facts"

 - "emotions"

- with strong constraints like:

 - No hallucinations

 - Only repeated traits

 - Must match stable long-term info


### 2. Personality Transformation
- Assistant Reply → rewrite via Gemini → New Tone Applied


- Each personality has explicit:

- Tone

- Speech rules

- Allowed expressions

- Forbidden patterns (e.g., “don’t add explanations”)


### 🔌 API Endpoints

- POST /extract-memory

- Extracts memory and updates the DB.
- Request
```
{
  "user_id": "user123",
  "messages": [
    "I love hiking.",
    "I usually go hiking on weekends.",
    "I'm working as a graphic designer."
  ]
}
````
- Response
```
{
  "status": "success",
  "memory": {
    "user_id": "user123",
    "preferences": ["hiking"],
    "facts": ["works as a graphic designer"],
    "emotions": [],
    "history_counts": { ... }
  }
}
```
- POST /transform-reply

- Rewrite a message into a selected personality.

- Request
```
{
  "base_reply": "You should try taking breaks during your work.",
  "personality": "calm_mentor"
}
```
- Response
```
{
  "before": "You should try taking breaks during your work.",
  "after": "Take a moment to step away now and then. It helps your mind settle.",
  "personality": "calm_mentor"
}
```

### test the application here
- Deployed on Render
- https://personality-ai-2.onrender.com/docs
