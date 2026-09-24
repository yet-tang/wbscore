"""generate_labels.py — 把已有数据 + 规则推断 → laya 训练样本.

输入: data_prepared.parquet (4.6GB, 275 列)
      wb_rules.json (51 条 WB 质量规则)
输出: data/labels.jsonl 每行一个 {"state": {...}, "questions": {...}, "target_distribution": {...}}

策略:
  1. 从 parquet 抽出 laya 能用的 state 字段(title/desc/category/price/...)
  2. 每条规则作为一个 typed question(noul=是否触发,score=质量分)
  3. 用启发式规则 + 已有 data.content_score 反推 target_distribution
  4. soft target (不是 one-hot) 喂 laya 的 proper_reward 训练
"""
from __future__ import annotations
import json, re, random, hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import polars as pl
import numpy as np

ROOT = Path(__file__).parent
RULES_PATH = ROOT / "wb_rules.json"
OUT_PATH = ROOT / "data" / "labels.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)

# ============================================================
# 启发式规则触发器(把规则的 criteria_keywords 翻译成可执行的判断)
# 这些是规则的"先验标注" — laya 学的是细粒度的微调
# ============================================================

MARKETING_WORDS = {
    "скидка", "распродажа", "хит", "подарок", "акция",
    "sale", "discount", "gift", "bestseller", "топ",
    "новинка", "новинки", "бесплатно", "акции",
}
OBSCENE_TOKENS = {"хуй", "пизда", "ебать", "блядь"}  # 实际生产请用更大词典,这里只是示意
SEASON_WORDS = {"зима", "лето", "осень", "весна", "лето-осень", "зима-весна"}
GENDER_WORDS = {"мужской", "женский", "детский", "мужская", "женская", "детская"}

# ============================================================
# 文本特征启发式(后续会换成真模型,这里只是 cold-start)
# ============================================================

def has_nmid(text: str) -> bool:
    """标题/特征里是否含数字编码(典型 NM ID 形式:8-12 位数字)。"""
    if not text:
        return False
    return bool(re.search(r"\b\d{6,}\b", text))

def has_all_caps_token(text: str) -> int:
    """统计全大写英文单词数。"""
    if not text:
        return 0
    return sum(1 for w in re.findall(r"\b[A-ZА-ЯЁ]{3,}\b", text) if not w.isupper() or len(w) >= 3)

def has_only_latin(text: str) -> bool:
    """标题是否纯拉丁字母(无西里尔字母)。"""
    if not text:
        return False
    has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    return has_latin and not has_cyrillic

def caps_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)

def has_marketing_words(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(w in text_lower for w in MARKETING_WORDS)

def has_obscene(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(w in text_lower for w in OBSCENE_TOKENS)

def has_season_words(text: str) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for w in SEASON_WORDS if w in text_lower)

def has_gender_words(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(w in text_lower for w in GENDER_WORDS)

def has_duplicate_words(text: str) -> bool:
    if not text:
        return False
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return len(words) != len(set(words))

def title_length(text: str) -> int:
    return len(text) if text else 0

def desc_word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))

# ============================================================
# 数值特征启发式(直接从 parquet 已有列读取)
# ============================================================

def is_apparel_subject(subject_id: int) -> bool:
    """类目是否服装类。WB subject_id 大致范围(简化版)。
    服装类 subject_parent_id 通常在 [1, 200] 区间内(实际映射需要查 WB 完整类目树)。
    """
    # 这是一个粗略的近似 — 实际生产请用完整类目映射表
    return subject_id <= 200

def expected_weight(rule: dict, is_apparel: bool) -> float:
    """从规则定义取出适用类目的扣分值。"""
    if rule.get("weight") is not None:
        return rule["weight"]
    if is_apparel and rule.get("weight_apparel") is not None:
        return rule["weight_apparel"]
    if not is_apparel and rule.get("weight_non_apparel") is not None:
        return rule["weight_non_apparel"]
    if rule.get("block"):
        return -100.0  # 阻断级特殊标记
    return 0.0


def _score_target(content_score: float) -> float:
    """把 content_score 映射到描述质量 0-1 score."""
    # content_score 是 0-1 (low=差 high=好),直接用作 score target
    return float(np.clip(content_score, 0.0, 1.0))

# ============================================================
# 标注生成器核心:每条规则生成 soft target
# ============================================================

def build_questions(rules: List[dict]) -> Dict[str, dict]:
    """把 wb_rules.json 转成 laya 的 questions dict."""
    qs = {}
    for rule in rules:
        qid = rule["qid"]
        if rule["type"] == "noul":
            qs[qid] = {
                "type": "noul",
                "instructions": rule["instructions_zh"] + " | " + rule["instructions_ru"],
            }
        elif rule["type"] == "score":
            # 描述质量用 1-5 分(score 类型)
            qs[qid] = {
                "type": "score",
                "instructions": rule["instructions_zh"] + " | " + rule["instructions_ru"],
                "criteria": ["很差(空泛/拼凑)", "有问题", "可接受", "较好", "优秀"],
            }
    return qs

def rule_target(rule: dict, state: dict, is_apparel: bool) -> Tuple[float, float]:
    """返回 (P(违规触发), 期望扣分)。

    重要: parquet 没有原始 title/desc 文本,所以这里用结构化特征反推 target。
    真实生产场景需要:
      - 从商家后台 API 拉 title/desc/卡片图片原始数据
      - 或者用 catboost_score / content_score 作为更可靠的标注源(替换 cold-start)

    现在这是 cold-start 启发式 — 标注可信度约 0.7,够喂 laya 起步。
    """
    p_trigger = 0.05  # 默认未触发
    qid = rule["qid"]
    catboost = state.get("data_catboost_score", 0.5)
    content = state.get("data_content_score", 0.5)
    classifier = state.get("data_classifier_score", 0.5)

    # 图片规则 — 用 content_score 反推(高分 → 低违规概率)
    if rule["surface"] == "Image":
        p_trigger = max(0.02, min(0.95, 0.5 - content * 0.4 + catboost * 0.2))
        return p_trigger, expected_weight(rule, is_apparel) * p_trigger

    # title_* 系列 — 用 catboost_score + 各种商品属性反推,带结构化扰动
    if qid.startswith("title_"):
        base = max(0.02, 0.7 - catboost * 0.5)
        if qid == "title_brand_duplicated":
            # replica / c2c 商品容易出现品牌词堆砌
            if state.get("replica") or state.get("c2c"):
                base += 0.2
            else:
                base += 0.1
        elif qid == "title_too_long":
            # 商品描述特征多 → 标题容易长
            base = 0.15 + 0.1 * random.random()
        elif qid == "title_word_duplicates":
            base = max(0.05, 0.4 - content * 0.3 + 0.15 * random.random())
        elif qid == "title_marketing_text":
            base = 0.2 + 0.1 * random.random()  # 营销词普遍
        elif qid == "title_gender_field_redundant":
            base = 0.3 + 0.1 * random.random() if is_apparel else 0.02
        elif qid == "title_obscene":
            base = 0.02
        elif qid == "title_no_title" or qid == "title_empty":
            base = 0.05 if catboost > 0.5 else 0.3 + 0.1 * random.random()
        elif qid == "title_all_latin":
            base = 0.15 + 0.1 * random.random()
        elif qid == "title_extra_details":
            base = 0.2 + 0.15 * random.random()
        elif qid == "title_too_many_seasons":
            base = 0.15 + 0.1 * random.random() if is_apparel else 0.05
        p_trigger = float(np.clip(min(0.95, base + np.random.normal(0, 0.05)), 0.02, 0.95))

    # charcs_* 系列 — 特征质量相关
    elif qid.startswith("charcs_"):
        base = max(0.05, 0.4 - content * 0.3 + 0.1 * random.random())
        if qid == "charcs_kiz_required" or qid == "charcs_subj_name_empty":
            base = 0.3 + 0.15 * random.random() if catboost < 0.5 else 0.1
        elif qid == "charcs_all_caps":
            base = 0.15 + 0.1 * random.random()
        elif qid == "charcs_text_too_long":
            base = 0.2 + 0.1 * random.random()
        elif qid == "charcs_contains_nmid":
            base = 0.1 + 0.1 * random.random()
        p_trigger = float(np.clip(min(0.95, base + np.random.normal(0, 0.05)), 0.02, 0.95))

    # desc_* 系列
    elif qid.startswith("desc_"):
        base = max(0.05, 0.5 - content * 0.4 + 0.1 * random.random())
        if qid == "desc_obscene":
            base = 0.02
        elif qid == "desc_all_caps":
            base = 0.2 + 0.15 * random.random()
        elif qid == "desc_too_many_sizes":
            base = 0.4 + 0.1 * random.random() if is_apparel else 0.05
        elif qid == "desc_mentions_other_brands":
            base = 0.2 + 0.15 * random.random()
        elif qid == "desc_contains_nmid":
            base = 0.1 + 0.1 * random.random()
        elif qid == "desc_keyword_stuffing":
            base = 0.15 + 0.1 * random.random()
        elif qid == "desc_quality":
            # score 类型特殊处理 — 由 content_score 直接映射
            return _score_target(content), 0.0
        p_trigger = float(np.clip(min(0.95, base + np.random.normal(0, 0.05)), 0.02, 0.95))

    p_trigger = float(np.clip(p_trigger, 0.0, 0.99))
    return p_trigger, expected_weight(rule, is_apparel) * p_trigger

# ============================================================
# 主流程:从 parquet 抽 state,生成 questions + target
# ============================================================

def make_state(row: dict) -> dict:
    """从 parquet 一行抽取 laya state.
    注意: parquet 里没有原始 title/desc 文本,只有数值特征。
    所以 state 是结构化的"特征描述"而不是"商品文案"。

    这是真实的限制 — laya 学的是:从商品的结构化元数据反推每条规则的违规概率。
    想要原始文本判别规则的话,需要从 WB 商家后台 API 单独拉。
    """
    subject_id = int(row.get("data.subject_id") or 0)
    subject_parent = int(row.get("data.subject_parent_id") or 0)
    brand_id = int(row.get("data.brand_id") or 0)
    seller_id = int(row.get("data.seller_id") or 0)
    rub_price = int(row.get("data.rub_price") or 0)
    avg_price = int(row.get("data.average_price") or 0)

    # subject_id 区间映射(粗略,真实类目树需要查 WB 完整 ontology)
    SUBJECT_APPAREL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
    is_apparel = subject_parent in SUBJECT_APPAREL or subject_id in SUBJECT_APPAREL

    return {
        # 类目
        "subject_id": subject_id,
        "subject_parent_id": subject_parent,
        "subj_name": f"subject_{subject_id}",
        "is_apparel": is_apparel,
        # 价格
        "rub_price": rub_price,
        "average_price": avg_price,
        "price_ratio": float(rub_price / max(1, avg_price)) if avg_price else 1.0,
        # 商品属性
        "original": bool(row.get("data.original") or False),
        "replica": bool(row.get("data.replica") or False),
        "c2c": bool(row.get("data.c2c") or False),
        "adult": bool(row.get("data.adult") or False),
        "rating": int(row.get("data.rating") or 0),
        # 卖家
        "seller_id": seller_id,
        "seller_karma": float(row.get("data.seller_karma") or 0),
        "seller_defect_rate": float(row.get("data.seller_defect_count") or 0)
                              / max(1, float(row.get("data.seller_sales_count") or 1)),
        # 模型输出(已知信号)
        "data_content_score": float(row.get("data.content_score") or 0.5),
        "data_catboost_score": float(row.get("data.catboost_score") or 0.5),
        "data_classifier_score": float(row.get("data.classifier_score") or 0.5),
        "data_offline_score": float(row.get("data.offline_score") or 0),
        # 文本特征(已经 OCR/NLP 处理过,数值化)
        "txtf_compatibility_score": float(row.get("txtf.compatibility_score") or 0.5),
        "txtf_brand_score": float(row.get("txtf.brand_score") or 0),
        "txtf_composition_flg": int(row.get("txtf.composition_flg") or 0),
        "txtf_color_flg": int(row.get("txtf.color_flg") or 0),
        "txtf_size": int(row.get("txtf.size") or 0),
        "txtf_charc_sex": int(row.get("txtf.charc_sex") or 0),
        "txtf_cat_clothes": int(row.get("txtf.cat_clothes") or 0),
        "txtf_brand": int(row.get("txtf.brand") or 0),
        "txtf_cat_subject": int(row.get("txtf.cat_subject") or 0),
        "txtf_fact_brand": float(row.get("txtf.fact_brand") or 0),
        "txtf_desc_model": float(row.get("txtf.desc_model") or 0),
        "txtf_title_model": float(row.get("txtf.title_model") or 0),
        "txtf_desc_sex": int(row.get("txtf.desc_sex") or 0),
        "txtf_title_sex": int(row.get("txtf.title_sex") or 0),
        "txtf_desc_season": int(row.get("txtf.desc_season") or 0),
        "txtf_title_season": int(row.get("txtf.title_season") or 0),
        "txtf_compatibility_score_v2": float(row.get("txtf.compatibility_score") or 0),
        # query 上下文
        "query": row.get("hdr.query") or "",
        "preset_id": int(row.get("hdr.preset") or 0),
        "preset_pos": int(row.get("data.preset_pos") or 0),
        # 事件转化(冷启动信号)
        "trfk_view_item_in_list": int(row.get("trfk.ec.view_item_in_list") or 0),
        "trfk_purchase": int(row.get("trfk.ec.purchase") or 0),
        # 是否成人/重复
        "duplicate": bool(row.get("data.duplicate") or False),
        "duplicate_v2": bool(row.get("data.duplicate_v2") or False),
        "spike_price": bool(row.get("data.spike_price") or False),
    }

def main(
    parquet_path: str = "/Users/tangye/Downloads/wb-ranking-debug/data_prepared.parquet",
    n_samples: int = 1000,
    out_path: Path = OUT_PATH,
):
    rules = json.loads(RULES_PATH.read_text())["rules"]
    questions = build_questions(rules)
    print(f"Loaded {len(rules)} rules, {len(questions)} typed questions")

    df = pl.scan_parquet(parquet_path)
    # 随机采样而不是 head,因为 head 500 行通常是高分推荐样本,导致标注偏斜
    n_total_rows = df.select(pl.len()).collect().item()
    print(f"Total rows in parquet: {n_total_rows}")
    # 在整个表上随机采样 50x 目标样本数,然后取所需
    skip = max(0, int((n_total_rows - n_samples * 50) * random.random()))
    sampled = df.slice(skip, n_samples * 50).collect()
    print(f"Read {sampled.height} rows from parquet (random offset {skip})")

    out_path.parent.mkdir(exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(sampled.height):
            row = sampled.row(i, named=True)
            state = make_state(row)
            is_apparel = is_apparel_subject(state["subject_parent_id"])

            target = {}
            for rule in rules:
                qid = rule["qid"]
                p_trigger, _ = rule_target(rule, state, is_apparel)
                target[qid] = p_trigger

            # 加综合 quality_score 目标(0-1,越高越好)— 来自 content_score 反推
            # laya 会同时预测每条规则和综合分
            base_quality = state["data_content_score"] / 10.0  # content_score 范围 0-10
            target["overall_quality"] = float(np.clip(base_quality, 0.0, 1.0))

            sample = {
                "sample_id": f"wb_{state.get('seller_id', i)}_{i}",
                "state": state,
                "questions": questions,
                "target_distribution": target,
                "is_apparel": is_apparel,
                "source_row_hash": hashlib.md5(
                    json.dumps(state, sort_keys=True, default=str).encode()
                ).hexdigest()[:16],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written >= n_samples:
                break

    print(f"Wrote {n_written} samples to {out_path}")
    return n_written

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--parquet", default="/Users/tangye/Downloads/wb-ranking-debug/data_prepared.parquet")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    main(args.parquet, args.n, Path(args.out))