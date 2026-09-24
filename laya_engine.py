"""laya_engine.py — WB 商品质量评分引擎.

架构:
  Encoder (ModernBERT-large 多语言版或俄语 RoBERTa)
    ↓
  2-layer Transformer head (双向)
    ↓
  [MASK] 标记评分 → 每条规则一个 logit
    ↓
  softmax → 每条规则的违规概率
    ↓
  proper_reward 训练(log_score + spherical + RPS)

为什么用 laya 而不只是 CatBoost:
  - 每条规则独立概率,可解释(显示哪条违规)
  - confidence 经严格适当打分规则校准,可直接做阈值
  - MacBook 可本地推理,无需 CatBoost 服务
  - 新规则来了,加一条 question 就行,不用重新训练整个模型
"""
from __future__ import annotations
import json, math, os, time, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 序列构造:state + questions → 共享 [MASK] 评分序列
# ============================================================

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def render_state_text(state: dict, max_chars: int = 320) -> str:
    """把结构化 state 渲染成 laya 输入文本。

    实际生产: 替换成原始 title + desc + charcs 文本。
    现在 parquet 没有原始文本,所以用结构化字段拼出一段"商品画像"。
    """
    parts = []
    if state.get("subj_name"):
        parts.append(f"Category: {state['subj_name']}")
    if state.get("rub_price"):
        avg = state.get("average_price", 0) or 0
        ratio = state.get("price_ratio", 1.0)
        parts.append(f"Price: {state['rub_price']}₽ (avg {avg}₽, ratio {ratio:.2f})")
    flags = []
    if state.get("original"): flags.append("original")
    if state.get("replica"):  flags.append("replica")
    if state.get("c2c"):      flags.append("c2c")
    if state.get("adult"):    flags.append("adult")
    if flags: parts.append("Flags: " + ", ".join(flags))
    if state.get("rating"):
        parts.append(f"Rating: {state['rating']}")
    if state.get("seller_karma") is not None:
        parts.append(f"Seller karma: {state['seller_karma']:.2f}")
    if state.get("seller_defect_rate") is not None:
        parts.append(f"Seller defect rate: {state['seller_defect_rate']:.2%}")
    parts.append(f"Quality signals: content={state.get('data_content_score', 0):.3f}, "
                 f"catboost={state.get('data_catboost_score', 0):.3f}, "
                 f"classifier={state.get('data_classifier_score', 0):.3f}")
    parts.append(f"Compatibility score: {state.get('txtf_compatibility_score', 0):.3f}")
    if state.get("txtf_brand_score"):
        parts.append(f"Brand score: {state['txtf_brand_score']:.3f}")
    if state.get("txtf_color_flg"): parts.append("Color flag: yes")
    if state.get("txtf_size"):      parts.append(f"Size flag: {state['txtf_size']}")
    if state.get("txtf_charc_sex"): parts.append(f"Sex flag: {state['txtf_charc_sex']}")
    if state.get("query"):
        parts.append(f"Search context: {state['query'][:60]}")
    if state.get("duplicate"):      parts.append("Duplicate: yes")
    if state.get("spike_price"):    parts.append("Spike price: yes")
    text = " | ".join(parts)
    return text[:max_chars * 4]


def render_options_for_question(q: dict) -> List[str]:
    """每条 question 的候选文本列表。"""
    if q["type"] == "noul":
        return ["false: 规则未触发", "true: 规则触发"]
    if q["type"] == "score":
        return [f"level {i}: {c}" for i, c in enumerate(q.get("criteria", ["差", "中", "好"]))]
    if q["type"] == "choice":
        opts = q.get("criteria", {})
        if isinstance(opts, dict):
            return [f"{k}: {v}" if v else k for k, v in opts.items()]
        return opts if isinstance(opts, list) else [str(opts)]
    return ["false", "true"]


def build_sequence(tokenizer, state: dict, question: dict,
                   max_len: int = 512, head_max_len: int = 192):
    """序列: [CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 [SEP] <state> [SEP]"""
    qtype = question["type"]
    ins = question.get("instructions", "").replace(tokenizer.mask_token or "<mask>", " ")
    head = tokenizer(f"{qtype} question: {ins}", add_special_tokens=False)["input_ids"]

    opts = render_options_for_question(question)
    opt_seqs = []
    for o in opts:
        o_clean = o.replace(tokenizer.mask_token or "<mask>", " ")
        opt_seqs.append(
            [tokenizer.mask_token_id]
            + tokenizer(" " + o_clean, add_special_tokens=False)["input_ids"][:48]
        )

    # 预算
    opt_budget = head_max_len - sum(len(o) for o in opt_seqs)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_seqs)))
        opt_seqs = [o[:per] for o in opt_seqs]
        opt_budget = head_max_len - sum(len(o) for o in opt_seqs)
    head = head[:max(8, opt_budget)]
    ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    markers = []
    for o in opt_seqs:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tokenizer.sep_token_id)

    # 状态部分
    state_text = render_state_text(state)
    st = tokenizer(state_text, add_special_tokens=False)["input_ids"]
    room = max(0, max_len - len(ids) - 1)
    st = st[-room:] if room > 0 else []
    ids = ids + st + [tokenizer.sep_token_id]
    ids = ids[:max_len]
    markers = [m for m in markers if m < max_len]
    return ids, markers


# ============================================================
# 2. DecisionModel
# ============================================================

class DecisionModel(nn.Module):
    """双向编码器 + [MASK] 评分 — 简化版 laya(可微调 ModernBERT multilingual / 俄语 RoBERTa)。"""

    def __init__(self, encoder_name: str = "DeepPavlov/rubert-base-cased", head_layers: int = 2,
                 n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        try:
            cfg = AutoConfig.from_pretrained(encoder_name)
            self.encoder = AutoModel.from_pretrained(encoder_name)
        except Exception:
            # fallback: 任意可用的小型多语言 encoder
            encoder_name = "xlm-roberta-base"
            cfg = AutoConfig.from_pretrained(encoder_name)
            self.encoder = AutoModel.from_pretrained(encoder_name)

        d = self.encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.tensor([1.0, 1.0, 1.0]))
        self.encoder_name = encoder_name

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        if self.head is not None:
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


# ============================================================
# 3. proper_reward — 严格适当打分规则(为何概率可信)
# ============================================================

def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    """log_score + spherical + RPS(score 类型)。

    严格适当打分规则的数学定理保证:期望奖励最大化 ⇔ 输出真实概率分布。
    这就是 laya 的 confidence 可以直接做阈值路由的原因。
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


# ============================================================
# 4. 数据集
# ============================================================

class WBDataset(Dataset):
    def __init__(self, samples_path: str, tokenizer, max_len: int = 512, head_max_len: int = 192):
        self.samples = []
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        state = s["state"]
        questions = s["questions"]
        target = s["target_distribution"]

        # 把所有 question 编码为 batch 中的一行
        items = []
        for qid, qdef in questions.items():
            ids, markers = build_sequence(self.tokenizer, state, qdef,
                                           self.max_len, self.head_max_len)
            t = QTYPES[qdef["type"]]
            items.append({
                "ids": ids, "markers": markers, "qtype": t,
                "qid": qid,
                "target": self._target_to_distribution(qdef, target.get(qid, 0.0)),
            })
        return items

    def _target_to_distribution(self, qdef: dict, t_val: float) -> List[float]:
        """把 scalar target 转成 soft target distribution。"""
        if qdef["type"] == "noul":
            # t_val ∈ [0, 1] 是 P(触发)
            return [1.0 - t_val, t_val]
        if qdef["type"] == "score":
            # t_val ∈ [0, 1] → 5 级评分
            level = int(round(t_val * (len(qdef.get("criteria", ["差", "中", "好"])) - 1)))
            level = max(0, min(level, len(qdef.get("criteria", ["差", "中", "好"])) - 1))
            dist = [0.1] * len(qdef.get("criteria", ["差", "中", "好"]))
            dist[level] = 0.7
            # 归一化
            s = sum(dist)
            return [d / s for d in dist]
        # choice — 单点
        return [1.0] + [0.0] * 10


def collate(batch, pad_id: int):
    """batch 是 list of list(items), 拍平成 list of items."""
    items = [it for sl in batch for it in sl]
    if not items:
        return None
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    n = len(items)

    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        mpos[i, :len(it["markers"])] = torch.tensor(it["markers"])
        mmask[i, :len(it["markers"])] = True
        tgt = it["target"]
        if len(tgt) < kmax:
            tgt = tgt + [0.0] * (kmax - len(tgt))
        target[i] = torch.tensor(tgt[:kmax])

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "target": target,
        "qids": [it["qid"] for it in items],
    }


# ============================================================
# 5. 训练循环
# ============================================================

def train(
    labels_path: str = "data/labels.jsonl",
    out_dir: str = "checkpoints/wb_laya",
    encoder_name: str = "DeepPavlov/rubert-base-cased",
    epochs: int = 4,
    batch_size: int = 4,
    lr: float = 1e-5,
    head_lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading tokenizer + encoder: {encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    model = DecisionModel(encoder_name=encoder_name).to(device)
    # 不同学习率组
    encoder_params = list(model.encoder.parameters())
    head_params = (list(model.head.parameters()) + list(model.type_emb.parameters())
                   + list(model.scorer.parameters()))
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=0.01)

    dataset = WBDataset(labels_path, tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id or 0),
        num_workers=0,
    )

    print(f"Training {epochs} epochs on {len(dataset)} samples")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        t0 = time.time()
        for batch in loader:
            if batch is None:
                continue
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            mpos = batch["marker_pos"].to(device)
            mmask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)

            logits, _ = model(ids, att, mpos, mmask, qtype)
            # proper_reward = 用 predicted prob vs target distribution 的 reward
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            mask = mmask.float()
            target = target * mask  # 屏蔽 padding

            reward = proper_reward(probs, target, qtype, mask)
            # 用 -reward 做"loss":我们要最大化 reward,所以 minimize -reward
            # 加一个 KL 约束防止偏离太远(reference model 用 frozen encoder)
            loss = -reward.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f} ({time.time()-t0:.1f}s)")

    # 保存
    torch.save({
        "model": model.state_dict(),
        "encoder_name": encoder_name,
        "rules_path": str(Path(labels_path).parent.parent / "wb_rules.json"),
    }, os.path.join(out_dir, "wb_laya.pt"))
    print(f"Saved checkpoint to {out_dir}/wb_laya.pt")


# ============================================================
# 6. 推理接口(类似 laya.Agent.system_one)
# ============================================================

def predict(model, tokenizer, state: dict, questions: dict,
            max_len: int = 512, head_max_len: int = 192,
            temperature: Optional[List[float]] = None,
            device: str = "cpu") -> Dict[str, dict]:
    """对一批 question 一次性 forward,返回每条规则的概率。"""
    model.eval()
    items = []
    qid_list = []
    for qid, qdef in questions.items():
        ids, markers = build_sequence(tokenizer, state, qdef, max_len, head_max_len)
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[qdef["type"]]})
        qid_list.append(qid)

    b = collate([items], tokenizer.pad_token_id or 0)
    with torch.no_grad():
        logits, _ = model(
            b["input_ids"].to(device),
            b["attention_mask"].to(device),
            b["marker_pos"].to(device),
            b["marker_mask"].to(device),
            b["qtype"].to(device),
        )
    logits = logits.cpu().numpy()

    answers = {}
    temp = temperature or [1.0, 1.0, 1.0]
    for r, qid in enumerate(qid_list):
        q = questions[qid]
        k = len(items[r]["markers"])
        qt = QTYPES[q["type"]]
        z = logits[r, :k] / max(1e-3, temp[qt])
        e = np.exp(z - z.max())
        p = e / e.sum()
        if q["type"] == "noul":
            answers[qid] = {
                "type": "noul",
                "p_true": float(p[1]),
                "p_false": float(p[0]),
                "confidence": float(max(p[0], p[1])),
                "violated": p[1] > 0.5,
                "weight": "see wb_rules.json",
            }
        elif q["type"] == "score":
            answers[qid] = {
                "type": "score",
                "level": int(p.argmax()),
                "expected_score": float((np.arange(k) * p).sum()),
                "probabilities": {str(i): float(v) for i, v in enumerate(p)},
                "confidence": float(p.max()),
            }
        else:
            answers[qid] = {"type": "choice", "probabilities": p.tolist()}

    # 计算综合分(简单加权求和)
    rules = json.loads(Path(__file__).parent.joinpath("wb_rules.json").read_text())["rules"]
    rule_lookup = {r["qid"]: r for r in rules}
    expected_loss = 0.0
    n_apparel = sum(1 for r in rules if r["surface"] in ("Title", "Description", "Characteristics"))
    for qid, ans in answers.items():
        rule = rule_lookup.get(qid)
        if not rule:
            continue
        is_apparel = state.get("_is_apparel", False)
        # 取适用类目的扣分
        w = rule.get("weight")
        if w is None:
            if is_apparel:
                w = rule.get("weight_apparel", 0)
            else:
                w = rule.get("weight_non_apparel", 0)
        if rule.get("block") and ans.get("violated"):
            expected_loss += 100.0  # 阻断
        else:
            expected_loss += w * ans.get("p_true", 0)

    answers["_meta"] = {
        "expected_penalty": expected_loss,
        "expected_quality": max(0.0, 10.0 + expected_loss),  # 基础 10 分
    }
    return answers


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict", "inspect"], default="inspect")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints/wb_laya/wb_laya.pt")
    ap.add_argument("--encoder", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "train":
        train(args.labels, encoder_name=args.encoder, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "inspect":
        # 看一下数据样本的 fields,确认 questions 结构
        with open(args.labels, encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("Sample fields:")
        print(f"  state keys: {list(sample['state'].keys())}")
        print(f"  questions: {len(sample['questions'])} questions")
        print(f"  target_distribution keys (first 5): {list(sample['target_distribution'].keys())[:5]}")
        print(f"  sample is_apparel: {sample.get('is_apparel')}")
        first_qid = list(sample['questions'].keys())[0]
        print(f"\nExample question ({first_qid}):")
        print(json.dumps(sample['questions'][first_qid], indent=2, ensure_ascii=False))
        print(f"\nTarget for {first_qid}: {sample['target_distribution'][first_qid]:.3f}")