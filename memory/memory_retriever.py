"""
Memory Retriever — Retrieves relevant memories before generating a response.
Implements recency bias and importance weighting for optimal recall.
"""

import time
from memory.memory_store import MemoryStore, MemoryType


class MemoryRetriever:
    """
    Retrieves and ranks relevant memories for injection into the conversation prompt.
    Combines semantic relevance with recency bias and importance weighting.
    """

    def __init__(self, store: MemoryStore, recency_weight: float = 0.3):
        """
        Args:
            store: The MemoryStore instance to search.
            recency_weight: How much to weight recency vs. relevance (0-1).
                           0 = pure semantic relevance, 1 = pure recency.
        """
        self.store = store
        self.recency_weight = recency_weight

    def retrieve(self, query: str, n_results: int = 5) -> list[str]:
        """
        Retrieve the most relevant memories for a given user message.
        Returns a list of memory content strings, ready for prompt injection.
        """
        # Fetch more candidates than needed for re-ranking
        candidates = self.store.search(query, n_results=n_results * 2)

        if not candidates:
            return []

        # Re-rank with combined scoring
        now = time.time()
        scored = []
        for mem in candidates:
            relevance = mem.get("relevance", 0)
            importance = mem.get("importance", 5) / 10.0
            timestamp = mem.get("timestamp", 0)

            # Recency score: exponential decay (half-life = 7 days)
            age_days = (now - timestamp) / 86400 if timestamp > 0 else 30
            recency = 2 ** (-age_days / 7)

            # Combined score
            final_score = (
                (1 - self.recency_weight) * relevance * importance +
                self.recency_weight * recency
            )

            scored.append({
                **mem,
                "final_score": final_score,
                "recency_score": recency,
            })

        # Sort by combined score and take top N
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_memories = scored[:n_results]

        return [m["content"] for m in top_memories]

    def retrieve_with_metadata(self, query: str, n_results: int = 5) -> list[dict]:
        """Same as retrieve() but returns full metadata for debugging/display."""
        candidates = self.store.search(query, n_results=n_results * 2)
        if not candidates:
            return []

        now = time.time()
        scored = []
        for mem in candidates:
            relevance = mem.get("relevance", 0)
            importance = mem.get("importance", 5) / 10.0
            timestamp = mem.get("timestamp", 0)
            age_days = (now - timestamp) / 86400 if timestamp > 0 else 30
            recency = 2 ** (-age_days / 7)
            final_score = (
                (1 - self.recency_weight) * relevance * importance +
                self.recency_weight * recency
            )
            scored.append({**mem, "final_score": round(final_score, 3), "age_days": round(age_days, 1)})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:n_results]
