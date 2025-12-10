from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Dict
import os

load_dotenv()
MONGO_DB_CONNECTION = os.getenv("MONGO_DB_CONNECTION")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

client = MongoClient(MONGO_DB_CONNECTION)
db = client[MONGO_DB_NAME]
collection = db["user_memory"]


def get_memory(user_id: str) -> Dict:
    """
    Return user's stored memory.
    If not exists, create an empty memory doc.
    """
    doc = collection.find_one({"user_id": user_id}, {"_id": 0})

    if not doc:
        empty = {
            "user_id": user_id,
            "preferences": [],
            "facts": [],
            "emotions": [],
            "history_counts": {}  # tracks how often a memory appears
        }
        collection.insert_one(empty)
        return empty

    return doc


def merge_memory(existing: dict, new: dict) -> dict:
    """
    Merge LLM extracted memory with existing memory.
    Adds new items and tracks frequency of repeated patterns.
    This prevents storing unstable or one-off traits.
    """

    history_counts = existing.get("history_counts", {})

    def update_field(field_name):
        updated = set(existing.get(field_name, []))

        for item in new.get(field_name, []):
            # Update frequency
            history_counts[item] = history_counts.get(item, 0) + 1

            # Only store if seen multiple times over time
            if history_counts[item] >= 2:
                updated.add(item)

        return sorted(list(updated))

    merged = {
        "user_id": existing.get("user_id"),
        "preferences": update_field("preferences"),
        "facts": update_field("facts"),
        "emotions": update_field("emotions"),
        "history_counts": history_counts
    }

    return merged


def update_memory(user_id: str, merged_memory: dict):
    """Write updated memory to MongoDB safely."""
    collection.update_one(
        {"user_id": user_id},
        {"$set": merged_memory},
        upsert=True
    )
