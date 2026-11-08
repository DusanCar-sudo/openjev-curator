#!/usr/bin/env python3
"""Compile the phase-2 fine-tune set from Keeper decisions + stamped proposals.

Every decision the owner makes on a stamped proposal is a labelled example in
their own judgement (curator-architecture phase 2). This compiler turns
the Keeper decisions directory (default /mnt/agentmem/keeper/decisions/*.json,
override with CURATOR_KEEPER) (gold) and proposals/*.json with a
model_opinion (pending candidates) into JSONL rows ready for classifier
training: premise, hypothesis, gold_label, model_probs, meta.

Gold label logic:
  accept  + supports_keeper   -> the model's label was RIGHT  (agree-positive)
  reject  + supports_keeper   -> the model's label was WRONG  (override)
  accept  + dissent           -> the dissent was RIGHT (model right to doubt)
  reject  + dissent           -> the dissent was WRONG (keeper was right)
Rows carry both views: 'model_correct' boolean and the raw probs, so either
target can be trained: reproduce Dusan's gate, or calibrate the NLI verdict.
"""
import json
import glob
import os
import sys

KEEPER = os.environ.get("CURATOR_KEEPER", "/mnt/agentmem/keeper")
OUT = os.environ.get("CURATOR_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))


def rows():
    for f in sorted(glob.glob(f"{KEEPER}/decisions/*.json")):
        d = json.load(open(f))
        mo = d.get("model_opinion")
        if not mo:
            continue
        yield {
            "split": "gold",
            "id": d["id"],
            "decision": d["decision"],
            "premise": mo.get("premise"),
            "hypothesis": mo.get("hypothesis"),
            "model_label": mo.get("label"),
            "model_probs": mo.get("probs"),
            "supports_keeper": mo.get("supports_keeper"),
            "model_correct": (d["decision"] == "accept") == bool(mo.get("supports_keeper")),
            "kind": d.get("kind"),
            "by": d.get("by"),
            "at": d.get("at"),
            "why": d.get("why", ""),
        }
    for f in sorted(glob.glob(f"{KEEPER}/proposals/*.json")):
        p = json.load(open(f))
        mo = p.get("model_opinion")
        if not mo:
            continue
        yield {
            "split": "pending",
            "id": p["id"],
            "decision": None,
            "premise": mo.get("premise"),
            "hypothesis": mo.get("hypothesis"),
            "model_label": mo.get("label"),
            "model_probs": mo.get("probs"),
            "supports_keeper": mo.get("supports_keeper"),
            "model_correct": None,
            "kind": p.get("kind"),
            "by": None,
            "at": mo.get("asked"),
            "why": "",
        }


def main():
    os.makedirs(OUT, exist_ok=True)
    out = list(rows())
    gold = [r for r in out if r["split"] == "gold"]
    path = os.path.join(OUT, "trainingset.jsonl")
    with open(path + ".tmp", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(f"{len(out)} rows -> {path}")
    print(f"  gold (decided): {len(gold)}   pending (awaiting owner): {len(out) - len(gold)}")
    if gold:
        right = sum(r["model_correct"] for r in gold)
        print(f"  model agreed with the owner's final call: {right}/{len(gold)}")


if __name__ == "__main__":
    main()
