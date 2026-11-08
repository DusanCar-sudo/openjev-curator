#!/usr/bin/env python3
"""Stamp model_opinion on Keeper proposals using the openjev NLI model.

Reads every pending proposal in the Keeper proposals directory (default
/mnt/agentmem/keeper/proposals/*.json, override with CURATOR_PROPOSALS), builds a
premise/hypothesis NLI pair, asks the model, writes the verdict back under
model_opinion. Proposals only -- notes and labels are never touched
(curator-architecture label firewall).

Pair construction: premise = what is TRUE (reality), hypothesis = what the
NOTE claims. contradiction => note conflicts with reality => keeper's call is
justified. entailment/neutral => model does not see the conflict.

Usage:
  .venv/bin/python opinion.py --dry   (print verdicts, write nothing)
  .venv/bin/python opinion.py         (stamp proposals in place)
"""
import glob
import json
import os
import re
import sys
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.environ.get("CURATOR_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/openjev-0.8b-nli-v2s-long"))
PROPOSALS = os.environ.get("CURATOR_PROPOSALS", "/mnt/agentmem/keeper/proposals")
LABELS = ["contradiction", "entailment", "neutral"]
# verdict that would justify the keeper's proposed action, per kind
SUPPORT = {"contradiction": "contradiction", "moved": "contradiction",
           "labels": "contradiction", "relink": "neutral"}


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


def verdict(tok, model, premise, hypothesis):
    text = f"Premise: {premise}\nHypothesis: {hypothesis}"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    probs = model(**inputs).logits.softmax(-1)[0]
    label = LABELS[int(probs.argmax())]
    return label, {k: round(float(v), 4) for k, v in zip(LABELS, probs)}


def pair_for(p):
    """(premise=reality, hypothesis=note's claim) from the proposal's own text."""
    kind, s = p["kind"], p["summary"]
    act = p.get("action", {})
    if kind in ("contradiction", "moved"):
        # "note says X but Y now" / "note names A, which is gone, but B exists"
        parts = re.split(r"\bbut\b", s, maxsplit=1)
        if len(parts) == 2:
            claim, reality = parts[0].strip(), parts[1].strip()
            # strip verdict clauses ("which is gone") so the hypothesis states
            # the note's claim, not the conclusion (label leak into training data)
            claim = re.sub(r",?\s*which is (gone|missing|stale|outdated)[,.]?\s*$", "", claim, flags=re.I).strip(" ,")
            return reality, claim
    if kind == "relink":
        old, new = act.get("old", "?"), act.get("new", "?")
        return (f"A note links to [[{old}]].",
                f"The link [[{old}]] in this note is broken or wrong; it should point to [[{new}]].")
    if kind == "labels":
        detail = p.get("detail") or s
        return detail[:1200], "Every label and verification status stated above is correct as written."
    return s, act.get("text", s)


def main():
    dry = any(a.startswith("--dry") for a in sys.argv[1:])
    tok, model = load()
    files = sorted(glob.glob(f"{PROPOSALS}/*.json"))
    print(f"{len(files)} proposals{' (dry run)' if dry else ''}")
    for f in files:
        p = json.load(open(f))
        if p.get("model_opinion"):
            print(f"  {p['id']}: already stamped, skip")
            continue
        premise, hypothesis = pair_for(p)
        t0 = time.time()
        label, probs = verdict(tok, model, premise, hypothesis)
        want = SUPPORT.get(kind := p["kind"], "neutral")
        op = {"model": "openjev-0.8b-nli-v2s-long", "label": label, "probs": probs,
              "supports_keeper": label == want, "ms": round((time.time() - t0) * 1000),
              "asked": time.strftime("%Y-%m-%dT%H:%M:%S"), "premise": premise,
              "hypothesis": hypothesis}
        if dry:
            print(f"  {p['id']}: {label} p={probs[label]} supports_keeper={label == want} (dry)")
            continue
        tmp = f + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(p | {"model_opinion": op}, fh, indent=1)
        os.replace(tmp, f)  # write-temp-then-rename
        print(f"  {p['id']}: {label} supports_keeper={label == want} stamped")


if __name__ == "__main__":
    main()
