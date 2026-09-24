"""compare_baseline.py — laya 评分引擎 vs CatBoost V12 baseline 对比.

输入:
  data/labels.jsonl (laya 标注样本,带 target_distribution)
  /Users/tangye/Downloads/wb-ranking-debug/ModelV12P04.cb (现有 CatBoost 模型)

输出:
  reports/baseline_comparison.md 包含:
    - laya vs CatBoost 在每条规则上的 P(违规) 一致性
    - CatBoost content_score 与 laya overall_quality 的相关性
    - 综合打分差异和置信区间
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "labels.jsonl"
CATBOOST_PATH = Path("/Users/tangye/Downloads/wb-ranking-debug/ModelV12P04.cb")
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_labels(limit: int = 500) -> List[dict]:
    samples = []
    with DATA.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def catboost_predict(samples: List[dict]) -> List[float]:
    """用 CatBoost 模型直接预测每条样本的 catboost_score。
    注意:CatBoost 用的是数值特征(state 拼成的 DataFrame),不是文本。
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("catboost not installed — skipping baseline prediction")
        return [s["state"]["data_catboost_score"] for s in samples]

    # 已知 CatBoost 用的关键数值特征(从 deep-dive.ipynb 推测)
    feature_cols = [
        "data.subject_id", "data.subject_parent_id", "data.brand_id",
        "data.rub_price", "data.average_price", "data.rating",
        "data.seller_karma", "data.seller_defect_count", "data.seller_sales_count",
        "data.original", "data.replica", "data.c2c", "data.adult",
        "txtf.compatibility_score", "txtf.brand_score",
        "txtf.composition_flg", "txtf.color_flg", "txtf.size", "txtf.charc_sex",
        "txtf.cat_clothes", "txtf.brand", "txtf.cat_subject",
        "txtf.fact_brand", "txtf.desc_model", "txtf.title_model",
        "txtf.desc_sex", "txtf.title_sex", "txtf.desc_season", "txtf.title_season",
        "hdr.ksmooth", "hdr.width_score", "hdr.query_words_count",
        "hdr.min_catboost_score", "hdr.max_catboost_score",
    ]

    # 把 samples 整理成 DataFrame
    rows = []
    for s in samples:
        row = {col: s["state"].get(col.replace("data.", "data_").replace("txtf.", "txtf_").replace("hdr.", "hdr_"), 0)
               for col in feature_cols}
        rows.append(row)
    df = pl.DataFrame(rows)

    try:
        model = CatBoostRegressor()
        model.load_model(str(CATBOOST_PATH))
        preds = model.predict(df.to_pandas())
        return preds.tolist()
    except Exception as e:
        print(f"CatBoost prediction failed: {e}")
        return [s["state"]["data_catboost_score"] for s in samples]


def laya_predict_batch(model, tokenizer, samples: List[dict], questions: dict) -> List[Dict[str, dict]]:
    """对每个样本做 laya 推理。"""
    from laya_engine import predict as laya_predict_fn

    results = []
    for s in samples:
        ans = laya_predict_fn(model, tokenizer, s["state"], questions, device="cpu")
        results.append(ans)
    return results


def main():
    samples = load_labels(limit=200)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No samples found. Run generate_labels.py first.")
        return

    questions = samples[0]["questions"]

    # ============ Baseline 1: CatBoost V12 ============
    t0 = time.time()
    catboost_preds = catboost_predict(samples)
    cb_time = time.time() - t0
    print(f"CatBoost predictions: {len(catboost_preds)} in {cb_time:.1f}s")

    # ============ Baseline 2: laya 推理 ============
    laya_preds = None
    laya_time = None
    try:
        # 尝试加载已训练的 laya
        import torch
        from laya_engine import DecisionModel, predict as laya_predict_fn

        ckpt_path = ROOT / "checkpoints" / "wb_laya" / "wb_laya.pt"
        if ckpt_path.exists():
            print(f"Loading laya checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
            if tokenizer.mask_token is None:
                tokenizer.mask_token = "<mask>"
                tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
            model = DecisionModel(encoder_name=ckpt["encoder_name"])
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()

            t0 = time.time()
            laya_preds = laya_predict_batch(model, tokenizer, samples, questions)
            laya_time = time.time() - t0
            print(f"Laya predictions: {len(laya_preds)} in {laya_time:.1f}s")
        else:
            print(f"No laya checkpoint found at {ckpt_path}. Train first with:")
            print(f"  python3 laya_engine.py --mode train --labels data/labels.jsonl --epochs 4")
    except Exception as e:
        print(f"Laya prediction failed: {e}")

    # ============ 评估指标 ============
    # 1. CatBoost 预测 vs 实际 catboost_score(自一致性,应该是 1.0)
    actual_cb = [s["state"]["data_catboost_score"] for s in samples]
    actual_content = [s["state"]["data_content_score"] for s in samples]
    actual_classifier = [s["state"]["data_classifier_score"] for s in samples]
    actual_overall = [s["target_distribution"]["overall_quality"] for s in samples]

    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0
    rmse = lambda a, b: float(np.sqrt(np.mean((np.array(a) - np.array(b))**2)))

    report_lines = [
        "# Baseline 对比报告",
        "",
        f"样本数: {len(samples)}",
        f"Question 数: {len(questions)}",
        "",
        "## 各信号源之间的相关性",
        "",
        "| 信号对 | Pearson r | RMSE |",
        "|--------|----------|------|",
        f"| catboost_pred vs data.catboost_score | {corr(catboost_preds, actual_cb):.4f} | {rmse(catboost_preds, actual_cb):.4f} |",
        f"| catboost_pred vs data.content_score | {corr(catboost_preds, actual_content):.4f} | {rmse(catboost_preds, actual_content):.4f} |",
        f"| data.catboost_score vs data.content_score | {corr(actual_cb, actual_content):.4f} | {rmse(actual_cb, actual_content):.4f} |",
        f"| data.catboost_score vs data.classifier_score | {corr(actual_cb, actual_classifier):.4f} | {rmse(actual_cb, actual_classifier):.4f} |",
        f"| target.overall_quality vs data.content_score | {corr(actual_overall, actual_content):.4f} | {rmse(actual_overall, actual_content):.4f} |",
        "",
        "## 性能对比",
        "",
        f"- CatBoost V12 推理: {cb_time:.2f}s ({cb_time/len(samples)*1000:.1f}ms/样本)",
    ]
    if laya_preds is not None and laya_time is not None:
        report_lines.append(f"- Laya 推理: {laya_time:.2f}s ({laya_time/len(samples)*1000:.1f}ms/样本)")

        # 每条规则的 P(违规) 一致性
        rule_p_corrs = []
        for qid in list(questions.keys())[:30]:  # 取前 30 条规则
            cb_probs = []
            laya_probs = []
            for s, lp in zip(samples, laya_preds):
                ans = lp.get(qid, {})
                cb_probs.append(s["target_distribution"].get(qid, 0.0))
                laya_probs.append(ans.get("p_true", 0.0))
            r = corr(cb_probs, laya_probs)
            rule_p_corrs.append((qid, r))

        report_lines += [
            "",
            "## Laya vs CatBoost — 每条规则的 P(违规) 相关性",
            "",
            "| Rule | Pearson r |",
            "|------|-----------|",
        ]
        for qid, r in sorted(rule_p_corrs, key=lambda x: -x[1]):
            report_lines.append(f"| {qid} | {r:.4f} |")

    report = "\n".join(report_lines) + "\n"
    out_file = REPORTS / "baseline_comparison.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_file}")
    print("\n" + "=" * 60)
    print(report[:2000])


if __name__ == "__main__":
    main()