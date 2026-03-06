#!/usr/bin/env python3
"""Run routing evaluation from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modality_router.eval import evaluate, load_jsonl
from modality_router.router import ModalityRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate modality router on JSONL data")
    parser.add_argument("--input", required=True, help="Path to JSONL file with query + label fields")
    parser.add_argument("--output", default="outputs/eval_report.json", help="Output report path")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    router = ModalityRouter()
    result = evaluate(rows, router)

    report = {
        "accuracy": result.accuracy,
        "total": result.total,
        "per_label_accuracy": result.per_label_accuracy,
        "confusion": result.confusion,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
