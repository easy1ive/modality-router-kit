# modality-router-kit

A lightweight, inspectable router that maps user queries to retrieval modalities for multimodal RAG pipelines.

## Why this project

Recent multimodal systems often use a fixed retrieval path. In practice, query intent varies a lot: some questions need text evidence, others need image grounding or temporal video understanding. This repo focuses on a practical router layer that can be tested and replaced easily.

## Industry context (2025-2026)

- General-purpose VLMs are getting stronger, but retrieval orchestration is still a major bottleneck.
- Router-first pipelines are becoming standard for latency and quality control in enterprise RAG stacks.
- Explainable routing decisions are increasingly important for debugging and model governance.

## Features

- Keyword-first routing with confidence score and traceable reasons.
- Simple labels: `no_rag`, `text`, `image`, `video`, `table`.
- Pure Python implementation without heavyweight runtime dependencies.
- Easy integration into any RAG orchestrator.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from modality_router import ModalityRouter

router = ModalityRouter()
print(router.predict("Describe the moment when the striker scores in the final video clip."))
```

Run a tiny offline evaluation:

```bash
python scripts/run_eval.py --input examples/queries.jsonl --output outputs/eval_report.json
```

## Routing schema

- `no_rag`: direct reasoning, rewrite, translation, arithmetic.
- `text`: paper/document-based evidence or world knowledge lookup.
- `image`: static visual understanding and attributes.
- `video`: temporal event and sequence understanding.
- `table`: structured lookup and comparison.

## Roadmap

- Add calibration on open multimodal QA benchmarks.
- Add optional embedding similarity router.
- Add more granular labels (`audio`, `chart`, `codebase`).
