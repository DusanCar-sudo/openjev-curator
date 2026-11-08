# openjev-curator

Decision model for the Mnemai Keeper pass-2 layer: openjev 0.8B NLI cross-encoder
(Qwen3.5-0.8B, `qwen3.5-0.8b-nli-v2s-long`) running on CPU.

## Why this model

- Trained cross-encoder for entailment / contradiction / neutral — the exact primitive
  for the curator's judgement calls (rival notes, contradiction detection, dedupe pairs).
- 853M params / 1.7 GB — fits the "up to 2B / 2 GB" rule and CPU-only, 12 GB RAM.
- One forward pass per option, nothing generated: deterministic, order-invariant
  (openjev reports 0/231 label changes under option shuffling), calibrated softmax.
- No token budget concern (curator-no-token-budget): ~1.3 s per call on this CPU is fine
  for a batched, async worker.

## Layout

- models/openjev-0.8b-nli-v2s-long/ — checkpoint (sha256 verified against HF)
- decide.py — CLI: one claim or a queue JSON file → verdict + probability
- test_nli.py — 6-case smoke test (5/6 pass)

## Use

    .venv/bin/python decide.py "note A text" "note B text"
    .venv/bin/python decide.py --queue queue.json

The nli_template from the checkpoint config is used verbatim:
"Premise: {premise}\nHypothesis: {hypothesis}".

Alternatives considered (pinggy.io comparison, 2026-09-23): kev (LoRA on Qwen3.5,
serves /v1/systemone, 5.4k stars), laya (ModernBERT 421M, fine-tune base, near-chance
zero-shot), SemIf (frozen Qwen3.5-4B, too big for CPU/RAM budget). openjev won on:
trained NLI head, size, zero-shot quality, CPU-only requirement.

## opinion.py — stamping pending proposals

    .venv/bin/python opinion.py --dry   # preview verdicts
    .venv/bin/python opinion.py         # write model_opinion into proposals

Reads /mnt/agentmem/keeper/proposals/*.json, builds a premise/reality vs
hypothesis/note-claim pair per kind (contradiction, moved, relink, labels),
asks the model, writes the verdict back into model_opinion. Label firewall
holds: proposals only, notes and labels never touched, Keeper never raises
a source/verified label from this.

supports_keeper = does the model's label back the Keeper's queued call:
- contradiction proposals: want "contradiction"
- moved: want "contradiction" (claim vs current path)
- relink: want "neutral" (a rename is not a semantic conflict)
- labels: want "contradiction" (dispute = labels not reliably as stated)

The accept/reject that follows (mnemai keeper accept/reject, Dusan's call at
the gate) is exactly the labelled example that builds the phase-2 fine-tune
set per curator-architecture.

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE). Free for personal,
research, educational, nonprofit and government use. Commercial use requires
a separate license from the author; open an issue to ask.
