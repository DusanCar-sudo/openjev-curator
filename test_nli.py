#!/usr/bin/env python3
"""Smoke test: openjev 0.8B NLI cross-encoder on curator-style judgements (CPU)."""
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "models/openjev-0.8b-nli-v2s-long"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

LABELS = ["contradiction", "entailment", "neutral"]


def nli(premise, hypothesis):
    text = f"Premise: {premise}\nHypothesis: {hypothesis}"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    t0 = time.time()
    out = model(**inputs)
    dt = (time.time() - t0) * 1000
    probs = out.logits.softmax(-1)[0]
    ranked = sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1])
    return ranked[0][0], ranked, dt


cases = [
    ("The server room is on the third floor.", "The server room is located above ground level.", "entailment"),
    ("The server room is on the third floor.", "The server room is in the basement.", "contradiction"),
    ("The server room is on the third floor.", "The server room has a coffee machine.", "neutral"),
    ("The archive is stored on disk-b.", "A note says the archive was deleted.", "contradiction"),
    ("GPU is an RTX 4090 with 24GB VRAM.", "The GPU has 24 gigabytes of memory.", "entailment"),
    ("The archive covers 2019 to 2024.", "The lead engineer prefers tea.", "neutral"),
]

print(f"{'case':>3} {'expected':<14} {'got':<14} {'p':>6} {'ms':>7}")
ok = 0
for i, (p, h, exp) in enumerate(cases):
    got, ranked, dt = nli(p, h)
    mark = "OK" if got == exp else "MISS"
    ok += got == exp
    print(f"{i:>3} {exp:<14} {got:<14} {ranked[0][1]:.3f} {dt:7.0f} {mark}")
print(f"\n{ok}/{len(cases)} correct")
