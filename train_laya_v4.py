#!/usr/bin/env python3
"""
Train laya v4 — class-balanced retraining with synthetic negatives.

Key fix vs v3: training data was heavily biased toward high-quality products,
so FAIL labels were rare (0% for obscene/other_brands/image rules, <2% for
most others). Model learned to default to "pass" and confidently hallucinated
violations at inference.

v4 adds 2700 synthetic products with DELIBERATE violations of every rule,
giving the model enough negative examples to actually learn what each FAIL
looks like.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

WORKSPACE = Path(__file__).parent
TRAIN_FILE = WORKSPACE / "data" / "training_set_v4.parquet"
WEIGHTS_DIR = WORKSPACE / "checkpoints" / "wb_laya_v4"
LOG_FILE = WORKSPACE / "training_log_v4.json"
RULES_FILE = WORKSPACE / "wb_rules.json"


def load_routes_data():
    import pyarrow.parquet as pq
    return pq.read_table(TRAIN_FILE).to_pylist()


def build_state(row):
    """State with ONLY observable features (no quality_score_proxy)."""
    return {
        "subject_id": row.get("subject_id", 0) or 0,
        "subject_parent_id": 1,
        "subj_name": (row.get("subj_name") or "").strip(),
        "is_apparel": bool(row.get("is_apparel") or "одежда" in (row.get("subj_root_name") or "").lower()),
        "rub_price": int((row.get("price_sale") or 0) * 100),
        "average_price": int((row.get("price_basic") or 0) * 100),
        "price_ratio": (row.get("price_sale") or 0) / max(row.get("price_basic") or 1, 1),
        "rating": float(row.get("rating") or 0),
        "feedbacks": int(row.get("feedbacks") or 0),
        "seller_karma": 0.7,
        # NOTE: removed data_content_score, data_catboost_score, data_classifier_score
        # These were quality_score_proxy / 100 — a leak. Use only observable features.
        "txtf_compatibility_score": 0.5,
        "txtf_brand_score": 1.0 if row.get("brand") else 0.0,
        "txtf_composition_flg": 1 if row.get("compositions_json") and row["compositions_json"] != "[]" else 0,
        "txtf_color_flg": 1,
        "txtf_size": 1 if (row.get("n_options") or 0) >= 5 else 0,
        "txtf_charc_sex": 1,
        "txtf_cat_clothes": 1 if row.get("is_apparel") else 0,
        "txtf_brand": 1 if row.get("brand") else 0,
        "txtf_description_len": int(row.get("description_len") or 0),
        "txtf_n_options": int(row.get("n_options") or 0),
        "txtf_photo_count": int(row.get("photo_count") or 0),
        "txtf_has_video": 1 if row.get("has_video") else 0,
    }


def check_rule(rule, row):
    code = rule["code"]
    cat = rule.get("surface", rule.get("category", "Image"))
    name = row.get("name") or ""
    desc = row.get("description") or ""
    desc_len = len(desc)
    options = json.loads(row.get("options_json") or "[]")
    comps = json.loads(row.get("compositions_json") or "[]")
    photos = int(row.get("photo_count") or 0)
    has_video = bool(row.get("has_video"))
    rating = float(row.get("rating") or 0)
    feedbacks = int(row.get("feedbacks") or 0)

    if "Img" in code or code.startswith("MessageBlur") or code.startswith("MessageWatermark") \
            or code.startswith("MessageToo") or code.startswith("MessageNotEnough"):
        if "NotEnough" in code:
            return 1.0 if photos >= 3 else 0.0
        if "TooSmall" in code:
            return 1.0 if photos >= 5 else 0.0
        return 1.0 if (photos >= 4 and (has_video or rating >= 4.0)) else 0.0

    if "Title" in code:
        if "MinLen" in code:
            return 1.0 if 5 <= len(name) <= 60 else 0.0
        if "MaxLen" in code:
            return 1.0 if len(name) <= 60 else 0.0
        if "NoCaps" in code:
            upper_ratio = sum(1 for c in name if c.isupper()) / max(len(name), 1)
            return 1.0 if upper_ratio < 0.5 else 0.0
        if "NoDigits" in code:
            digit_ratio = sum(1 for c in name if c.isdigit()) / max(len(name), 1)
            return 1.0 if digit_ratio < 0.3 else 0.0
        if "Brand" in code:
            return 1.0 if row.get("brand") else 0.0
        return 1.0 if len(name) >= 5 else 0.0

    if "Состав" in code or "Composition" in code or code.startswith("MessageTxtf"):
        return 1.0 if comps else 0.0

    if "Description" in code and "Title" not in code:
        return 1.0 if row.get("description_len", 0) >= 200 else 0.0

    if "Charc" in code or "Charact" in code or "Option" in code:
        return 1.0 if len(options) >= 3 else 0.0

    if "Color" in code:
        return 1.0 if any("Цвет" in str(o.get("name", "")) for o in options) else 0.0

    if "Size" in code and "Img" not in code:
        return 1.0 if len(options) >= 5 else 0.0

    if "Desc" in code:
        if "MinLen" in code or "Len" in code:
            return 1.0 if desc_len >= 200 else 0.0
        if "MaxLen" in code:
            return 1.0 if desc_len <= 5000 else 0.0
        if "Link" in code or "Url" in code:
            return 0.0 if ("http://" in desc or "https://" in desc or "www." in desc) else 1.0
        if "Phone" in code:
            return 0.0 if any(c.isdigit() for c in desc.replace(" ", "")) and len(desc) < 100 else 1.0
        return 1.0 if desc_len >= 200 else 0.0

    if desc_len >= 200 and row.get("brand") and photos >= 3:
        return 1.0
    return 0.0


def main():
    print("Loading data...")
    rows = load_routes_data()
    print(f"  {len(rows)} products")

    with open(RULES_FILE) as f:
        rules_data = json.load(f)
    rules = rules_data.get("rules", [])
    print(f"  {len(rules)} rules")

    # Pick representative rules
    target_codes = set()
    by_cat = {}
    for rule in rules:
        c = rule.get("surface", rule.get("category", "Image"))
        by_cat.setdefault(c, []).append(rule)
    target_per_cat = {"Image": 4, "Title": 4, "Characteristics": 4, "Description": 4}
    for cat, n in target_per_cat.items():
        for rule in by_cat.get(cat, [])[:n]:
            target_codes.add(rule["code"])
    print(f"  using {len(target_codes)} rules")

    print("Building sequences...")
    sequences = []
    for r in rows:
        state = build_state(r)
        questions = {}
        targets = {}
        for rule in rules:
            if rule["code"] not in target_codes:
                continue
            try:
                p = check_rule(rule, r)
            except Exception:
                p = 0.5
            target = [1.0 - p, p]
            questions[rule["code"]] = {
                "type": "choice",
                "category": rule.get("surface", rule.get("category", "Image")),
                "instructions": rule.get("instructions_zh", rule.get("description", ""))[:200],
                "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
            }
            targets[rule["code"]] = target
        sequences.append({
            "state": state,
            "questions": questions,
            "target_distribution": targets,
        })

    print("\nLoading laya model...")
    sys.path.insert(0, str(WORKSPACE))
    from laya_engine import DecisionModel, build_sequence, QTYPES, proper_reward
    from transformers import AutoTokenizer

    encoder_name = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    model = DecisionModel(encoder_name=encoder_name)
    device = "cpu"
    model = model.to(device)

    enc_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters()) + list(model.type_emb.parameters()) + list(model.scorer.parameters())
    optim = torch.optim.AdamW([
        {"params": enc_params, "lr": 2e-5},
        {"params": head_params, "lr": 2e-4},
    ], weight_decay=0.01)

    epochs = int(os.environ.get("LAYA_EPOCHS", "3"))
    batch_size = int(os.environ.get("LAYA_BATCH", "8"))
    log = {"loss": [], "n_samples": len(sequences), "epochs": epochs,
           "encoder": encoder_name, "started_at": time.time(), "version": "v4 (class-balanced)"}

    print(f"Training {epochs} epochs on {len(sequences)} samples (no leakage features)")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        np.random.shuffle(sequences)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        n = 0

        for batch_start in range(0, len(sequences), batch_size):
            batch = sequences[batch_start:batch_start + batch_size]
            optim.zero_grad()
            batch_loss = 0.0
            n_items = 0

            for seq in batch:
                state = seq["state"]
                rule_items = []
                targets_list = []
                for qid, qdef in seq["questions"].items():
                    target = torch.tensor(seq["target_distribution"][qid], dtype=torch.float32)
                    if target.sum() == 0:
                        target[0] = 0.01
                    ids, markers = build_sequence(tokenizer, state, qdef, max_len=384, head_max_len=128)
                    if len(markers) < 2:
                        continue
                    rule_items.append((ids, markers))
                    targets_list.append(target[:len(markers)])
                if not rule_items:
                    continue

                max_len = max(len(ids) for ids, _ in rule_items)
                max_markers = max(len(m) for _, m in rule_items)

                ids_batch = torch.zeros((len(rule_items), max_len), dtype=torch.long)
                mask_batch = torch.zeros((len(rule_items), max_len), dtype=torch.long)
                mpos_batch = torch.zeros((len(rule_items), max_markers), dtype=torch.long)
                mmask_batch = torch.zeros((len(rule_items), max_markers), dtype=torch.bool)
                target_batch = torch.zeros((len(rule_items), max_markers), dtype=torch.float32)
                qtype_batch = torch.zeros((len(rule_items),), dtype=torch.long)

                for i, ((ids, markers), target) in enumerate(zip(rule_items, targets_list)):
                    ids_batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    mask_batch[i, :len(ids)] = 1
                    mpos_batch[i, :len(markers)] = torch.tensor(markers, dtype=torch.long)
                    mmask_batch[i, :len(markers)] = True
                    target_batch[i, :len(target)] = target
                    qtype_batch[i] = QTYPES["choice"]

                ids_batch = ids_batch.to(device)
                mask_batch = mask_batch.to(device)
                mpos_batch = mpos_batch.to(device)
                mmask_batch = mmask_batch.to(device)
                target_batch = target_batch.to(device)
                qtype_batch = qtype_batch.to(device)
                m_float = mmask_batch.float()
                target_batch = target_batch * m_float

                logits, _ = model(ids_batch, mask_batch, mpos_batch, mmask_batch, qtype_batch)
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                reward = proper_reward(probs, target_batch, qtype_batch, m_float)
                loss = -reward.mean()
                batch_loss = batch_loss + loss
                n_items += 1

            if n_items > 0:
                loss_val = batch_loss / n_items
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                epoch_loss += loss_val.item()
                n_batches += 1
                n += n_items

            if n_batches % 100 == 0 and n_batches > 0:
                elapsed = time.time() - t0
                rate = n / max(elapsed, 0.1)
                eta = (len(sequences) * epochs - (epoch * len(sequences) + n)) / max(rate, 0.1)
                print(f"  epoch {epoch+1}  batch {n_batches}  loss={epoch_loss / n_batches:.4f}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        log["loss"].append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  ({elapsed/60:.1f}min)")

        torch.save({
            "model": model.state_dict(),
            "encoder_name": encoder_name,
            "rules_path": str(RULES_FILE),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }, WEIGHTS_DIR / "wb_laya.pt")
        print(f"  saved → {WEIGHTS_DIR}/wb_laya.pt")

    log["finished_at"] = time.time()
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Log → {LOG_FILE}")
    print(f"Final → {WEIGHTS_DIR}/wb_laya.pt")


if __name__ == "__main__":
    main()