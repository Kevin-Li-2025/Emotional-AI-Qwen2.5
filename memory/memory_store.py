"""
Memory Store — Persistent long-term memory using ChromaDB.
Stores three types of memories: Facts, Emotional Events, and Preferences.
Supports semantic search for retrieving relevant memories during conversation.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("[WARNING] chromadb not installed. Run: pip install chromadb")


class MemoryType(str, Enum):
    FACT = "fact"             # "User's favorite food is ramen"
    EMOTION = "emotion"       # "User insulted Lin Xia on 2026-04-08"
    PREFERENCE = "preference" # "User prefers playful tone over formal"


@dataclass
class Memory:
    content: str
    memory_type: MemoryType
    importance: float = 5.0       # 1-10 scale
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    timestamp: float = 0.0
    metadata: dict = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {},
        }


class MemoryStore:
    """
    ChromaDB-backed persistent memory store for Lin Xia.
    Memories persist across sessions and are retrieved via semantic search.
    """

    def __init__(self, db_path: str = "./memory_db", collection_name: str = "linxia_memories"):
        self.db_path = db_path
        self.collection_name = collection_name

        if not HAS_CHROMADB:
            self.client = None
            self.collection = None
            return

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
        )

    def add_memory(self, memory: Memory) -> str:
        """Add a memory to the store. Returns the memory ID."""
        if not self.collection:
            return ""

        memory_id = f"mem_{int(memory.timestamp * 1000)}"

        self.collection.add(
            documents=[memory.content],
            metadatas=[{
                "memory_type": memory.memory_type.value,
                "importance": memory.importance,
                "emotional_valence": memory.emotional_valence,
                "timestamp": memory.timestamp,
                **(memory.metadata or {}),
            }],
            ids=[memory_id]
        )

        return memory_id

    def search(self, query: str, n_results: int = 5,
               memory_type: MemoryType = None,
               min_importance: float = 0.0) -> list[dict]:
        """
        Semantic search for relevant memories.
        Optionally filter by memory type and minimum importance.
        """
        if not self.collection:
            return []

        # Build where filter
        where_filter = {}
        if memory_type:
            where_filter["memory_type"] = memory_type.value
        if min_importance > 0:
            where_filter["importance"] = {"$gte": min_importance}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

            memories = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    memories.append({
                        "content": doc,
                        "relevance": round(1 - distance, 3),  # Convert distance to similarity
                        "type": meta.get("memory_type", "unknown"),
                        "importance": meta.get("importance", 0),
                        "timestamp": meta.get("timestamp", 0),
                    })

            return memories

        except Exception as e:
            print(f"[MEMORY SEARCH ERROR] {e}")
            return []

    def get_all_memories(self, limit: int = 100) -> list[dict]:
        """Retrieve all stored memories (for debugging/display)."""
        if not self.collection:
            return []

        results = self.collection.get(limit=limit)
        memories = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                memories.append({"content": doc, **meta})
        return memories

    def get_memory_count(self) -> int:
        """Return total number of stored memories."""
        if not self.collection:
            return 0
        return self.collection.count()

    def clear(self):
        """Delete all memories (use with caution)."""
        if self.client:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
