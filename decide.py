#!/usr/bin/env python3
"""openjev decision CLI for the Mnemai curator pass 2.

Reads judgement calls as JSON, writes verdicts as JSON. Never touches the notes:
verdicts go to proposals/, per curator-architecture (label firewall).

Input (stdin or file arg), one call per line or a JSON array:
  {"id": "contradiction-67392d26", "kind": "contradiction",
   "premise": "<text of note A>", "hypothesis": "<text of note B>"}

Output per call:
  {"id": ..., "kind": ..., "label": "contradiction|entailment|neutral",
   "probs": {...}, "action": "propose|clear"}

Actions:
  contradiction -> propose (notes conflict, needs the owner or a merge proposal)
  entailment    -> propose (near-duplicate pair, merge candidate)
  neutral       -> clear   (unrelated, drop from queue)
"""
import json
import sys
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/openjev-0.8b-nli-v2s-long"
LABELS = ["contradiction", "entailment", "neutral"]
ACTION = {"contradiction": "propose", "entailment": "propose", "neutral": "clear"}


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


def verdict(tok, model, premise, hypothesis):
    text = f"Premise: {premise}\nHypothesis: {hypothesis}"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    probs = model(**inputs).logits.softmax(-1)[0].tolist()
    ranked = sorted(zip(LABELS, probs), key=lambda x: -x[1])
    return ranked[0][0], dict(zip(LABELS, (round(p, 4) for p in probs)))


def main():
    raw = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    calls = [json.loads(l) for l in raw.splitlines() if l.strip()] if raw.lstrip().startswith("{") else json.loads(raw)
    if isinstance(calls, dict):
        calls = [calls]
    tok, model = load()
    out = []
    for c in calls:
        t0 = time.time()
        label, probs = verdict(tok, model, c["premise"], c["hypothesis"])
        out.append({
            "id": c.get("id"), "kind": c.get("kind", "nli"),
            "label": label, "probs": probs, "action": ACTION[label],
            "model": "openjev-0.8b-nli-v2s-long",
            "ms": round((time.time() - t0) * 1000),
        })
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
