#!/usr/bin/env python3
"""Aggregate the 5 jsonl files for a single model into one summary.json.

Usage:
    python3 scripts/summarize_eval.py --eval-dir results/<model_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarise_pages(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("status") == "ok"]
    if not scored:
        return {"n": 0}
    n = len(scored)
    sum_p = sum(r["precision"] for r in scored)
    sum_r = sum(r["recall"] for r in scored)
    sum_f1 = sum(r["f1"] for r in scored)
    tp = sum(r["tp"] for r in scored)
    fp = sum(r["fp"] for r in scored)
    fn = sum(r["fn"] for r in scored)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) else 0.0)
    return {
        "n": n,
        "macro": {"precision": sum_p / n, "recall": sum_r / n, "f1": sum_f1 / n},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "tp": tp, "fp": fp, "fn": fn},
    }


def summarise_labels(rows: list[dict], label_field: str = "label") -> dict:
    counts = Counter()
    for r in rows:
        if r.get("status") != "ok":
            continue
        lbl = r.get(label_field)
        if lbl:
            counts[lbl] += 1
    total = sum(counts.values())
    return {"n": total, "by_label": dict(counts)}


def summarise_answers(rows: list[dict]) -> dict:
    judged = [r for r in rows if isinstance(r.get("consistent"), bool)]
    n = len(judged)
    correct = sum(1 for r in judged if r["consistent"])
    has_tag = sum(1 for r in rows if r.get("has_answer_tag"))
    return {
        "n_judged": n,
        "n_correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "n_with_answer_tag": has_tag,
        "n_total": len(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, required=True)
    args = p.parse_args()
    d = args.eval_dir

    summary = {
        "eval_dir": str(d),
        "infer": {"n": len(load_jsonl(d / "infer.jsonl"))},
        "pages": summarise_pages(load_jsonl(d / "pages.jsonl")),
        "bbox": summarise_labels(load_jsonl(d / "bbox.jsonl")),
        "facts": summarise_labels(load_jsonl(d / "facts.jsonl")),
        "answer": summarise_answers(load_jsonl(d / "answer.jsonl")),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
