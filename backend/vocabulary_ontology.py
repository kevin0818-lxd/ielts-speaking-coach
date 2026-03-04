import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except Exception:  # pragma: no cover - graceful fallback if dependency is absent
    nx = None


class _SimpleDiGraph:
    """Tiny fallback graph used when networkx is unavailable."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict] = {}
        self._edges: Dict[str, Dict[str, Dict]] = {}

    def add_node(self, node_id: str, **attrs) -> None:
        existing = self._nodes.get(node_id, {})
        existing.update(attrs)
        self._nodes[node_id] = existing

    def add_edge(self, src: str, dst: str, **attrs) -> None:
        self._edges.setdefault(src, {})
        existing = self._edges[src].get(dst, {})
        existing.update(attrs)
        self._edges[src][dst] = existing
        self.add_node(src)
        self.add_node(dst)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self, data: bool = False):
        return self._nodes.items() if data else self._nodes.keys()

    def neighbors(self, node_id: str):
        return self._edges.get(node_id, {}).keys()

    def get_edge_data(self, src: str, dst: str) -> Optional[Dict]:
        return self._edges.get(src, {}).get(dst)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(len(v) for v in self._edges.values())


class VocabularyOntology:
    """
    Ontology wrapper for concept-aware recommendation expansion.
    Uses networkx.DiGraph when available; otherwise uses a minimal fallback.
    """

    def __init__(
        self,
        context_vocab: Dict,
        context_keywords: Dict[str, List[str]],
        seed_path: Optional[str] = None,
    ) -> None:
        self._graph = nx.DiGraph() if nx is not None else _SimpleDiGraph()
        self._context_vocab = context_vocab or {}
        self._context_keywords = context_keywords or {}
        self._seed_path = seed_path
        self._topic_nodes: Set[str] = set()
        self._register_nodes: Set[str] = set()
        self._discourse_nodes: Set[str] = set()
        self._word_nodes: Set[str] = set()
        self._build_graph()

    @staticmethod
    def _normalize_word(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    @staticmethod
    def _word_node(word: str) -> str:
        return f"word:{word}"

    @staticmethod
    def _topic_node(topic: str) -> str:
        return f"topic:{topic}"

    @staticmethod
    def _register_node(register: str) -> str:
        return f"register:{register}"

    @staticmethod
    def _discourse_node(tag: str) -> str:
        return f"discourse:{tag}"

    def _add_word_node(self, word: str) -> str:
        norm = self._normalize_word(word)
        node = self._word_node(norm)
        self._graph.add_node(node, kind="word", text=norm)
        self._word_nodes.add(node)
        return node

    def _build_graph(self) -> None:
        # Topic nodes from keyword map
        for topic, keywords in self._context_keywords.items():
            topic_key = self._normalize_word(topic)
            topic_node = self._topic_node(topic_key)
            self._graph.add_node(topic_node, kind="topic", text=topic_key)
            self._topic_nodes.add(topic_node)
            for kw in keywords:
                kw_node = self._add_word_node(kw)
                self._graph.add_edge(kw_node, topic_node, relation="fits_context")
                self._graph.add_edge(topic_node, kw_node, relation="context_keyword")

        # Word, register, and context relations from context vocab
        for source_word, mapping in self._context_vocab.items():
            source_node = self._add_word_node(source_word)
            for ctx, options in (mapping or {}).items():
                if ctx == "default":
                    discourse_node = self._discourse_node("default")
                    self._graph.add_node(discourse_node, kind="discourse", text="default")
                    self._discourse_nodes.add(discourse_node)
                    self._graph.add_edge(source_node, discourse_node, relation="fits_context")
                else:
                    topic_node = self._topic_node(self._normalize_word(ctx))
                    if not self._graph.has_node(topic_node):
                        self._graph.add_node(topic_node, kind="topic", text=self._normalize_word(ctx))
                        self._topic_nodes.add(topic_node)
                    self._graph.add_edge(source_node, topic_node, relation="fits_context")

                if not isinstance(options, list):
                    continue
                for candidate in options:
                    if not isinstance(candidate, dict):
                        continue
                    cword = self._normalize_word(str(candidate.get("word", "")))
                    if not cword:
                        continue
                    cnode = self._add_word_node(cword)
                    self._graph.add_edge(source_node, cnode, relation="semantic_neighbor")
                    self._graph.add_edge(cnode, source_node, relation="semantic_neighbor")

                    reg = self._normalize_word(str(candidate.get("reg", "neutral")))
                    reg_node = self._register_node(reg)
                    self._graph.add_node(reg_node, kind="register", text=reg)
                    self._register_nodes.add(reg_node)
                    self._graph.add_edge(cnode, reg_node, relation="register")
                    self._graph.add_edge(reg_node, cnode, relation="register_member")

                    if ctx != "default":
                        topic_node = self._topic_node(self._normalize_word(ctx))
                        self._graph.add_edge(cnode, topic_node, relation="fits_context")

                    if " " in cword:
                        colloc_tag = self._discourse_node("collocation")
                        self._graph.add_node(colloc_tag, kind="discourse", text="collocation")
                        self._discourse_nodes.add(colloc_tag)
                        self._graph.add_edge(cnode, colloc_tag, relation="collocates_with")

        self._load_seed_edges()

    def _load_seed_edges(self) -> None:
        if not self._seed_path:
            return
        if not os.path.exists(self._seed_path):
            return
        try:
            with open(self._seed_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        prereqs = payload.get("prerequisites", [])
        collocations = payload.get("collocations", [])
        semantic_links = payload.get("semantic_neighbors", [])

        for row in prereqs:
            src = self._add_word_node(str(row.get("from", "")))
            dst = self._add_word_node(str(row.get("to", "")))
            if src and dst:
                self._graph.add_edge(src, dst, relation="prerequisite_of")

        for row in collocations:
            src = self._add_word_node(str(row.get("from", "")))
            dst = self._add_word_node(str(row.get("to", "")))
            if src and dst:
                self._graph.add_edge(src, dst, relation="collocates_with")

        for row in semantic_links:
            src = self._add_word_node(str(row.get("from", "")))
            dst = self._add_word_node(str(row.get("to", "")))
            if src and dst:
                self._graph.add_edge(src, dst, relation="semantic_neighbor")
                self._graph.add_edge(dst, src, relation="semantic_neighbor")

    def map_text_to_concepts(self, text: str, part: int = 1) -> List[str]:
        text_lower = (text or "").lower()
        concepts: Set[str] = set()
        for topic, keywords in self._context_keywords.items():
            for kw in keywords:
                kw_norm = self._normalize_word(kw)
                if kw_norm and kw_norm in text_lower:
                    concepts.add(topic.lower())
                    break
        if not concepts:
            # lightweight fallback from explicit context vocab keys seen in text
            for source in self._context_vocab.keys():
                s = self._normalize_word(source)
                if s and s in text_lower:
                    concepts.add("general")
                    break
        if not concepts:
            concepts.add("general")
        if part == 3:
            concepts.add("abstract_discussion")
        return sorted(concepts)

    def get_candidate_neighbors(
        self,
        concepts: List[str],
        max_hops: int = 2,
        register_hint: str = "neutral",
    ) -> List[str]:
        if max_hops < 1:
            max_hops = 1
        register_hint = self._normalize_word(register_hint or "neutral")
        concept_nodes: List[str] = []
        for c in concepts or []:
            topic_node = self._topic_node(self._normalize_word(c))
            if self._graph.has_node(topic_node):
                concept_nodes.append(topic_node)

        results: Set[str] = set()
        frontier: List[Tuple[str, int]] = [(n, 0) for n in concept_nodes]
        visited: Set[str] = set(concept_nodes)

        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            for nb in self._graph.neighbors(node):
                if nb in visited:
                    continue
                visited.add(nb)
                frontier.append((nb, depth + 1))
                nb_data = None
                if nx is not None:
                    nb_data = self._graph.nodes[nb]
                else:
                    for nid, attrs in self._graph.nodes(data=True):
                        if nid == nb:
                            nb_data = attrs
                            break
                if not nb_data:
                    continue
                if nb_data.get("kind") == "word":
                    text = str(nb_data.get("text", "")).strip()
                    if text:
                        results.add(text)

        # Register-aware ordering: words connected to requested register first.
        reg_node = self._register_node(register_hint)
        preferred: List[str] = []
        fallback: List[str] = []
        for w in sorted(results):
            wn = self._word_node(w)
            has_reg = False
            if self._graph.has_node(wn) and self._graph.has_node(reg_node):
                edge = self._graph.get_edge_data(wn, reg_node)
                has_reg = bool(edge)
            if has_reg:
                preferred.append(w)
            else:
                fallback.append(w)
        return preferred + fallback

    def get_prerequisite_chain(self, node: str) -> List[str]:
        target = self._word_node(self._normalize_word(node))
        if not self._graph.has_node(target):
            return []
        chain: List[str] = []
        current = target
        seen: Set[str] = set()
        while current not in seen:
            seen.add(current)
            parent = None
            # Find incoming prerequisite_of edges
            if nx is not None:
                for src, _, attrs in self._graph.in_edges(current, data=True):
                    if attrs.get("relation") == "prerequisite_of":
                        parent = src
                        break
            else:
                for src, _ in self._graph.nodes(data=True):
                    attrs = self._graph.get_edge_data(src, current)
                    if attrs and attrs.get("relation") == "prerequisite_of":
                        parent = src
                        break
            if not parent:
                break
            if parent.startswith("word:"):
                chain.append(parent.split("word:", 1)[1])
            current = parent
        chain.reverse()
        return chain

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": int(self._graph.number_of_nodes()),
            "edges": int(self._graph.number_of_edges()),
            "word_nodes": len(self._word_nodes),
            "topic_nodes": len(self._topic_nodes),
            "register_nodes": len(self._register_nodes),
            "discourse_nodes": len(self._discourse_nodes),
        }
