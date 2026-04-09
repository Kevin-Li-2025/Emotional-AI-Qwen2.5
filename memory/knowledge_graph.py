"""
Knowledge Graph Memory — Structured Entity-Relationship Memory
Uses NetworkX to maintain a persistent graph of entities and relationships
extracted from conversations with the user.

Unlike flat vector search (ChromaDB), this provides:
  1. Structured traversal: "User → has pet → 豆豆 → is a → dog → likes → bones"
  2. Causal chains: "User insulted Lin Xia → caused → hurt → led to → cold shoulder"
  3. Contextual recall: Related entities are pulled in together, not as fragments

Graph is persisted to JSON for durability across sessions.
"""

import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Node & Edge types
# ---------------------------------------------------------------------------

class NodeType:
    PERSON = "person"
    PLACE = "place"
    OBJECT = "object"
    EVENT = "event"
    EMOTION = "emotion"
    PREFERENCE = "preference"
    TIME = "time"
    TRAIT = "trait"

class EdgeType:
    HAS = "has"
    LIKES = "likes"
    DISLIKES = "dislikes"
    FELT = "felt"
    AT = "at"
    RELATED_TO = "related_to"
    CAUSED = "caused"
    IS_A = "is_a"
    SAID = "said"
    REMEMBERS = "remembers"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"


class KnowledgeGraph:
    """
    Persistent knowledge graph using NetworkX.
    Stores entities (nodes) and relationships (edges) extracted from conversations.
    """

    def __init__(self, persist_path: str = "./memory_db/knowledge_graph.json"):
        self.persist_path = persist_path
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from JSON file if it exists."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                print(f"[KG] Loaded {self.graph.number_of_nodes()} nodes, "
                      f"{self.graph.number_of_edges()} edges")
            except Exception as e:
                print(f"[KG] Failed to load graph: {e}. Starting fresh.")
                self.graph = nx.DiGraph()
        else:
            print("[KG] No existing graph found. Starting fresh.")

    def _save(self):
        """Persist graph to JSON."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------

    def add_entity(self, name: str, node_type: str, properties: dict = None) -> str:
        """
        Add an entity node to the graph.
        If entity already exists, merge properties.
        """
        node_id = self._normalize_id(name)

        if self.graph.has_node(node_id):
            # Merge properties
            existing = self.graph.nodes[node_id]
            if properties:
                existing.update(properties)
            existing["last_updated"] = time.time()
        else:
            self.graph.add_node(node_id, **{
                "label": name,
                "type": node_type,
                "created_at": time.time(),
                "last_updated": time.time(),
                **(properties or {}),
            })

        self._save()
        return node_id

    def add_relation(self, source: str, relation: str, target: str,
                     properties: dict = None) -> bool:
        """
        Add a directed relationship edge between two entities.
        Creates source/target nodes if they don't exist.
        """
        src_id = self._normalize_id(source)
        tgt_id = self._normalize_id(target)

        # Auto-create nodes if needed
        if not self.graph.has_node(src_id):
            self.add_entity(source, NodeType.OBJECT)
        if not self.graph.has_node(tgt_id):
            self.add_entity(target, NodeType.OBJECT)

        self.graph.add_edge(src_id, tgt_id, **{
            "relation": relation,
            "created_at": time.time(),
            **(properties or {}),
        })

        self._save()
        return True

    # -------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------

    def query_related(self, entity: str, depth: int = 2) -> list[dict]:
        """
        Find all entities related to the given entity within N hops.
        Returns a list of {entity, relation, connected_to, properties} dicts.
        """
        node_id = self._normalize_id(entity)
        if not self.graph.has_node(node_id):
            return []

        results = []
        visited = set()

        def _traverse(current: str, current_depth: int):
            if current_depth > depth or current in visited:
                return
            visited.add(current)

            # Outgoing edges
            for _, neighbor, edge_data in self.graph.out_edges(current, data=True):
                neighbor_data = self.graph.nodes.get(neighbor, {})
                results.append({
                    "from": self.graph.nodes[current].get("label", current),
                    "relation": edge_data.get("relation", "related_to"),
                    "to": neighbor_data.get("label", neighbor),
                    "to_type": neighbor_data.get("type", "unknown"),
                })
                _traverse(neighbor, current_depth + 1)

            # Incoming edges
            for predecessor, _, edge_data in self.graph.in_edges(current, data=True):
                pred_data = self.graph.nodes.get(predecessor, {})
                results.append({
                    "from": pred_data.get("label", predecessor),
                    "relation": edge_data.get("relation", "related_to"),
                    "to": self.graph.nodes[current].get("label", current),
                    "to_type": self.graph.nodes[current].get("type", "unknown"),
                })
                _traverse(predecessor, current_depth + 1)

        _traverse(node_id, 0)

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = (r["from"], r["relation"], r["to"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def query_by_type(self, node_type: str) -> list[dict]:
        """Find all entities of a given type."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == node_type:
                results.append({"id": node_id, **data})
        return results

    def to_context_string(self, query: str = "", max_items: int = 15) -> str:
        """
        Serialize the most relevant parts of the graph into a context string
        suitable for injection into the LLM system prompt.

        If query is provided, prioritize entities matching the query.
        Otherwise, return the most recently updated entities.
        """
        if self.graph.number_of_nodes() == 0:
            return ""

        # Find relevant nodes
        if query:
            # Fuzzy match: find nodes whose labels contain query keywords
            keywords = set(query.lower().replace("？", "").replace("。", "").split())
            scored_nodes = []
            for node_id, data in self.graph.nodes(data=True):
                label = data.get("label", "").lower()
                score = sum(1 for kw in keywords if kw in label or label in kw)
                if score > 0:
                    scored_nodes.append((node_id, score))
            scored_nodes.sort(key=lambda x: -x[1])
            target_nodes = [n[0] for n in scored_nodes[:5]]
        else:
            # Fallback: most recently updated
            sorted_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("last_updated", 0),
                reverse=True
            )
            target_nodes = [n[0] for n in sorted_nodes[:5]]

        if not target_nodes:
            return ""

        # Gather relationships for target nodes
        all_relations = []
        for node_id in target_nodes:
            related = self.query_related(node_id, depth=1)
            all_relations.extend(related)

        # Deduplicate and format
        seen = set()
        lines = ["[Knowledge Graph — What Lin Xia Remembers]"]
        for r in all_relations[:max_items]:
            key = (r["from"], r["relation"], r["to"])
            if key not in seen:
                seen.add(key)
                lines.append(f"  {r['from']} → {r['relation']} → {r['to']}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def to_mermaid(self, max_nodes: int = 30) -> str:
        """Export graph as Mermaid diagram for Gradio display."""
        if self.graph.number_of_nodes() == 0:
            return "graph LR\n  Empty[No memories yet]"

        lines = ["graph LR"]
        node_count = 0

        # Type → shape mapping
        shape_map = {
            NodeType.PERSON: ('["', '"]'),    # Rectangle
            NodeType.EMOTION: ('(("', '"))'),  # Circle
            NodeType.PREFERENCE: ('{"', '"}'), # Diamond
            NodeType.EVENT: ('["', '"]'),
        }

        added_nodes = set()
        for src, tgt, data in self.graph.edges(data=True):
            if node_count >= max_nodes:
                break

            src_data = self.graph.nodes[src]
            tgt_data = self.graph.nodes[tgt]
            src_label = src_data.get("label", src)
            tgt_label = tgt_data.get("label", tgt)
            relation = data.get("relation", "→")

            # Sanitize for Mermaid
            src_safe = src.replace(" ", "_").replace("-", "_")
            tgt_safe = tgt.replace(" ", "_").replace("-", "_")
            src_label_safe = src_label.replace('"', "'")
            tgt_label_safe = tgt_label.replace('"', "'")

            s_open, s_close = shape_map.get(src_data.get("type"), ('["', '"]'))
            t_open, t_close = shape_map.get(tgt_data.get("type"), ('["', '"]'))

            lines.append(f'  {src_safe}{s_open}{src_label_safe}{s_close} -->|{relation}| {tgt_safe}{t_open}{tgt_label_safe}{t_close}')
            node_count += 1

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(
                sorted(
                    {t: sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == t)
                     for t in set(d.get("type", "?") for _, d in self.graph.nodes(data=True))}.items()
                )
            ),
        }

    @staticmethod
    def _normalize_id(name: str) -> str:
        """Normalize entity name to a consistent ID."""
        return name.strip().lower().replace(" ", "_")

    def clear(self):
        """Clear the entire graph."""
        self.graph.clear()
        self._save()


if __name__ == "__main__":
    # Demo
    kg = KnowledgeGraph(persist_path="./memory_db/knowledge_graph.json")

    # Add some sample entities and relations
    kg.add_entity("用户", NodeType.PERSON, {"alias": "Kevin"})
    kg.add_entity("林夏", NodeType.PERSON, {"role": "AI companion"})
    kg.add_entity("豆豆", NodeType.OBJECT, {"species": "dog"})
    kg.add_entity("草莓蛋糕", NodeType.OBJECT, {"category": "food"})
    kg.add_entity("薰衣草", NodeType.OBJECT, {"color": "purple"})

    kg.add_relation("用户", EdgeType.HAS, "豆豆", {"since": "childhood"})
    kg.add_relation("豆豆", EdgeType.IS_A, "柯基犬")
    kg.add_relation("用户", EdgeType.LIKES, "草莓蛋糕")
    kg.add_relation("用户", EdgeType.LIKES, "薰衣草", {"reason": "奶奶家后院种了很多"})
    kg.add_relation("林夏", EdgeType.FELT, "开心", {"trigger": "用户买了草莓蛋糕"})
    kg.add_relation("林夏", EdgeType.FELT, "受伤", {"trigger": "用户说她是人工智能"})

    print(f"\nGraph stats: {kg.get_stats()}")
    print(f"\n--- Query: 用户 (depth=2) ---")
    for r in kg.query_related("用户", depth=2):
        print(f"  {r['from']} → {r['relation']} → {r['to']}")

    print(f"\n--- Context string ---")
    print(kg.to_context_string("用户喜欢什么"))

    print(f"\n--- Mermaid ---")
    print(kg.to_mermaid())
