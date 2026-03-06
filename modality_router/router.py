"""Keyword-first modality router for multimodal RAG."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re


@dataclass(frozen=True)
class RouteDecision:
    """Single routing output."""

    label: str
    confidence: float
    reasons: List[str]


class ModalityRouter:
    """Route user queries to a likely retrieval modality.

    Labels:
    - ``no_rag``: likely answerable without external retrieval.
    - ``text``: requires text/knowledge retrieval.
    - ``image``: requires visual understanding over static images.
    - ``video``: requires temporal understanding.
    - ``table``: structured/tabular lookup and comparison.
    """

    LABELS: Tuple[str, ...] = ("no_rag", "text", "image", "video", "table")

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.keyword_weights: Dict[str, List[Tuple[str, float]]] = {
            "image": [
                ("image", 1.0),
                ("photo", 1.0),
                ("picture", 1.0),
                ("appearance", 0.8),
                ("color", 0.7),
                ("object", 0.6),
                ("diagram", 0.8),
            ],
            "video": [
                ("video", 1.0),
                ("clip", 1.0),
                ("moment", 0.9),
                ("sequence", 0.8),
                ("motion", 0.8),
                ("frame", 0.7),
            ],
            "table": [
                ("table", 1.0),
                ("csv", 0.9),
                ("spreadsheet", 0.9),
                ("rows", 0.6),
                ("columns", 0.6),
                ("compare", 0.5),
            ],
            "text": [
                ("paper", 0.7),
                ("document", 0.9),
                ("article", 0.7),
                ("report", 0.6),
                ("evidence", 0.7),
                ("citation", 0.8),
                ("source", 0.6),
            ],
            "no_rag": [
                ("calculate", 0.7),
                ("math", 0.7),
                ("translate", 0.6),
                ("rewrite", 0.6),
            ],
        }

    def predict(self, query: str) -> RouteDecision:
        if not query or not query.strip():
            return RouteDecision(label="no_rag", confidence=0.99, reasons=["empty_query"])

        text = query.lower().strip()
        scores = {label: 0.0 for label in self.LABELS}
        reasons: List[str] = []

        for label, pairs in self.keyword_weights.items():
            for keyword, weight in pairs:
                if keyword in text:
                    scores[label] += weight
                    reasons.append(f"{label}:{keyword}")

        if re.search(r"\b(why|how|what|which|who|when|where)\b", text):
            scores["text"] += 0.3

        if re.search(r"\b\d+[\s]*[\*\/+\-][\s]*\d+\b", text):
            scores["no_rag"] += 0.8
            reasons.append("no_rag:equation")

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score < self.threshold:
            if text.endswith("?"):
                best_label = "text"
                reasons.append("fallback:text_question")
                best_score = max(best_score, 0.45)
            else:
                best_label = "no_rag"
                reasons.append("fallback:no_rag")
                best_score = max(best_score, 0.45)

        confidence = round(best_score / (best_score + 1.0), 4)
        return RouteDecision(label=best_label, confidence=confidence, reasons=reasons or ["weak_signal"])

    def predict_many(self, queries: List[str]) -> List[RouteDecision]:
        """Batch-friendly wrapper for lightweight routing jobs."""
        return [self.predict(query) for query in queries]
