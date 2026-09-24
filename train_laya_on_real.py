#!/usr/bin/env python3
"""
Train laya scoring model on real WB product data.

Loads:
  data/training_set.parquet (built by build_training_set.py)

Trains:
  - ModernBERT-large encoder (already wrapped in laya)
  - 2-layer transformer head
  - Score head (regression to quality_score_proxy)

Outputs:
  laya_weights.pt — serialized model
  training_log.json — loss curves
"""
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent
TRAIN_FILE = WORKSPACE / "data" / "training_set.parquet"
WEIGHTS_FILE = WORKSPACE / "laya_weights.pt"
LOG_FILE = WORKSPACE / "training_log.json"


def main():
    if not TRAIN_FILE.exists() and not TRAIN_FILE.with_suffix(".jsonl").exists():
        sys.exit(f"missing {TRAIN_FILE} — run build_training_set.py first")
    if not TRAIN_FILE.exists():
        # jsonl fallback
        TRAIN_FILE = TRAIN_FILE.with_suffix(".jsonl")

    print("Loading training data...")
    if str(TRAIN_FILE).endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(TRAIN_FILE)
            rows = table.to_pylist()
        except ImportError:
            sys.exit("install pyarrow to read parquet: pip install pyarrow")
    else:
        rows = []
        with open(TRAIN_FILE) as f:
            for line in f:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows")

    if len(rows) < 100:
        print(f"Warning: only {len(rows)} rows; need more data for meaningful training")

    # Build laya sequence: state + questions + target_distribution
    # state: features (subject, brand, supplier, prices)
    # questions: rule codes to evaluate (Message21, Message12, Message31, Message41, ...)
    # target_distribution: probability that each rule passes (derived from quality_score_proxy)
    print("\nBuilding laya sequences...")
    sequences = []
    for r in rows:
        # Encode features as text tokens (laya uses ModernBERT)
        state = (
            f"subject: {r['subj_name']} | brand: {r['brand']} | supplier: {r['supplier']} | "
            f"price: {r['price_sale']:.0f}₽ | rating: {r['rating']:.1f} | feedbacks: {r['feedbacks']}"
        )
        # Combine title + description + characteristics into context
        context = (
            f"Title: {r['name']}\n"
            f"Description: {r['description'][:500]}\n"
            f"Options: {r['options_json'][:300]}\n"
            f"Photos: {r['photo_count']}, Video: {r['has_video']}"
        )
        # target: regression on quality_score_proxy
        target = r["quality_score_proxy"] / 100.0  # normalize to [0,1]
        sequences.append({
            "nm_id": r["nm_id"],
            "state": state,
            "context": context,
            "target": target,
        })

    # Try to actually train. Falls back to rule-based scorer if laya unavailable.
    print("\nTraining...")
    try:
        # Import laya from its submodule
        sys.path.insert(0, str(WORKSPACE))
        from laya_engine import DecisionModel, train_step, build_sequence
        import torch

        model = DecisionModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        n_epochs = 3
        log = {"loss": [], "epochs": n_epochs}

        for epoch in range(n_epochs):
            total = 0.0
            for i, seq in enumerate(sequences):
                # Build a laya sequence
                tokens = build_sequence(seq["state"], seq["context"])
                pred = model(tokens)
                loss = (pred - seq["target"]) ** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg = total / len(sequences)
            log["loss"].append(avg)
            print(f"  epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

        torch.save(model.state_dict(), WEIGHTS_FILE)
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nSaved → {WEIGHTS_FILE}")
        print(f"Training log → {LOG_FILE}")
    except ImportError as e:
        print(f"\nlaya_engine not installed ({e}); skipping training.")
        print("Install with: pip install laya  (or run from this workspace)")
        # Save sequences as a checkpoint so we can train later
        seq_file = WORKSPACE / "data" / "laya_sequences.jsonl"
        with open(seq_file, "w") as f:
            for s in sequences:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(sequences)} sequences → {seq_file}")


if __name__ == "__main__":
    main()