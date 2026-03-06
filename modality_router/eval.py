"""Evaluation helpers for routing datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .router import ModalityRouter


@dataclass
class EvalResult:
    accuracy: float
    total: int
    per_label_accuracy: Dict[str, float]
    confusion: Dict[str, Dict[str, int]]


def load_jsonl(path: str | Path) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate(items: Iterable[dict], router: ModalityRouter | None = None) -> EvalResult:
    router = router or ModalityRouter()
    total = 0
    correct = 0

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_total: Dict[str, int] = defaultdict(int)
    label_correct: Dict[str, int] = defaultdict(int)

    for item in items:
        query = item["query"]
        gold = item["label"]
        pred = router.predict(query).label

        total += 1
        label_total[gold] += 1
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1
            label_correct[gold] += 1

    per_label_accuracy = {
        label: round(label_correct[label] / count, 4) if count else 0.0
        for label, count in sorted(label_total.items())
    }
    accuracy = round(correct / total, 4) if total else 0.0

    normalized_confusion = {
        gold: {pred: count for pred, count in sorted(preds.items())}
        for gold, preds in sorted(confusion.items())
    }

    return EvalResult(
        accuracy=accuracy,
        total=total,
        per_label_accuracy=per_label_accuracy,
        confusion=normalized_confusion,
    )
