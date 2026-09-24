"""
Generate actionable "how to fix" recommendations for triggered WB rules.

Each recommendation is a concrete, vendor-doable suggestion based on
which rule fired + the actual product state. Designed for WB sellers
who want to know what to edit, not just what's wrong.

Usage:
    from recommend import build_recommendations
    recs = build_recommendations(triggered_rule_codes, state)
"""
from __future__ import annotations
from typing import Iterable


# Per-rule-code fix advice (Chinese, vendor-actionable)
# Keys are rule codes from wb_rules.json; values are {category, fix}
# fix is a function that takes the state and returns a specific string.
RULE_FIXES = {
    # === Image rules ===
    "MessageTooSmallItem": "主图里商品占比太小 — 把商品摆满画面 80%以上面积。",
    "MessageImgBadMultipleObjects": "一张图里放了多个商品 — 每张图只展示一个商品或一个颜色。",
    "MessageCollage": "图片是拼接/拼贴 — 用单张高清实拍图，不要海报式合成。",
    "MessageImgBadCrop": "图片裁切有问题（边缘被切、构图歪）— 重新构图，确保商品完整。",
    "MessageTooSmallItem_main": "主图尺寸不够 — 至少 900×1200 px，纯白/浅色背景。",
    "MessageWatermark": "图片上有水印/文字 — 去掉所有非商品本身的文字、logo、水印。",
    "MessageNotEnoughImages": "图片数量不足 — 至少上传 5-7 张：正面、背面、细节、包装、场景。",
    "MessageNoVideo": "没有商品视频 — 上传 15-30 秒展示视频，能显著提升转化。",
    # === Title rules ===
    "MessageTitleNMID": "标题里有数字（疑似货号）— 标题只描述商品，不要放 SKU/编号。",
    "MessageTitleUpperWord": "标题全大写英文 — 改用俄语/常规格式，ALL CAPS 会扣分。",
    "MessageWordDuplicates": "标题有重复词 — 删除重复字词，标题要简洁不冗余。",
    "MessageTitleMarketingText": "标题有营销词（'лучший', 'топ', 'акция'）— 删掉营销词，只留商品描述。",
    "MessageObsceneWord": "标题含违禁词 — 立即检查并替换（俄文+英文敏感词都要查）。",
    "MessageTitleMinLen": "标题太短 — 至少 5 个字、含品牌 + 商品类型 + 关键属性。",
    "MessageTitleMaxLen": "标题过长 — 控制在 60 字符以内。",
    "MessageTitleNoCaps": "标题大写比例过高 — 大写字母 ≤ 50%。",
    "MessageTitleNoDigits": "标题数字过多 — 数字应只用于型号/规格，不是营销数字。",
    "MessageTitleBrand": "标题缺少品牌 — 如果有品牌一定要写，没有就别编。",
    "MessageNoTitle": "标题为空 — 标题是必填项，至少写 5 个字。",
    # === Characteristics (options/composition) rules ===
    "MessageMissingCharcs": "缺少关键规格参数 — 在特性里补全：颜色、尺寸、材质、性别等。",
    "MessageBadCharcLength": "规格值过长或过短 — 规格值控制在 2-50 字符之间。",
    "MessageBadPackSize": "包装/数量信息异常 — 检查'Количество в упаковке'字段是否合理。",
    "MessageBadRUSize": "俄罗斯尺码表错误 — 用俄罗斯尺码表（RU），不要用 US/EU 混着写。",
    "MessageTxtfNoComposition": "缺少 Состав 字段 — 服装/纺织品类必须有 Состав: 材料+百分比。",
    # === Description rules ===
    "MessageAnotherBrands": "描述里出现其他品牌名 — 删掉所有非本品牌的提及，避免纠纷。",
    "MessageDescrNMID": "描述里有货号/数字串 — 描述只写商品介绍，不要放 SKU。",
    "MessageDescrUpperWord": "描述全大写 — 正常俄文书写，不要 ALL CAPS。",
    "MessageDescrMinLen": "描述太短 — 至少 200 字，写材质、用途、保养、尺码建议。",
    "MessageDescrMaxLen": "描述过长 — 控制在 5000 字符以内。",
    "MessagePoorDescription": "描述质量差 — 加具体细节：材质、工艺、适用场景、保养方式。",
    "MessageDescLink": "描述里含链接 — 删掉 http://、www. 等外链，WB 禁止站外引流。",
    "MessageDescPhone": "描述里有电话号码 — 删掉电话/微信/Telegram，违反 WB 规则。",
}


def _state_driven_fix(code: str, state: dict) -> str | None:
    """Generate a specific fix string based on actual state when possible."""
    if not state:
        return None

    # Title length
    if code == "MessageTitleMinLen" and state.get("name"):
        n = len(state["name"])
        if n < 5:
            return f"标题只有 {n} 字 — 至少补到 5 个字以上（含品牌+品类+关键属性）。"
    if code == "MessageTitleMaxLen" and state.get("name"):
        n = len(state["name"])
        return f"标题 {n} 字 — 删减到 60 字以内，去掉冗余修饰。"

    # Description length
    if code == "MessageDescrMinLen" and state.get("description") is not None:
        n = len(state["description"] or "")
        need = max(0, 200 - n)
        return f"描述 {n} 字 — 至少补到 200 字（还差 {need} 字）。建议写材质/工艺/适用场景/保养方式。"

    # Photo count
    if code in ("MessageNotEnoughImages", "MessageTooSmallItem") and state.get("photo_count") is not None:
        pc = state["photo_count"]
        return f"目前 {pc} 张图 — 至少 5-7 张：正面/背面/细节/包装/场景。"

    # Composition
    if code == "MessageTxtfNoComposition" and not state.get("compositions"):
        return "Состав 字段为空 — 服装类必须有'Состав: 材料 百分比'（例：хлопок 95%, полиэстер 5%）。"

    # Options count
    if code == "MessageMissingCharcs" and state.get("n_options") is not None:
        n = state.get("n_options", 0)
        return f"当前 {n} 个规格 — 至少补 3 个：颜色、尺寸、性别（或材质/适用人群）。"

    return None


def build_recommendations(
    triggered: Iterable,
    state: dict | None = None,
    max_n: int = 8,
) -> list[dict]:
    """Build a deduplicated, prioritized list of actionable fixes.

    Args:
        triggered: list of RuleHit dicts (each has 'code', 'penalty', 'category')
        state: optional product state dict for state-specific advice
        max_n: max recommendations to return (top-N by penalty)

    Returns:
        list of {code, category, penalty, fix} dicts
    """
    seen = set()
    recs = []
    for r in triggered:
        code = r.get("code") if isinstance(r, dict) else getattr(r, "code", None)
        if not code or code in seen:
            continue
        seen.add(code)
        fix = _state_driven_fix(code, state or {}) or RULE_FIXES.get(code) or (
            "查看 WB 官方规则说明并修正该字段。"
        )
        recs.append({
            "code": code,
            "category": r.get("category") if isinstance(r, dict) else getattr(r, "category", ""),
            "penalty": r.get("penalty") if isinstance(r, dict) else getattr(r, "penalty", 0),
            "fix": fix,
        })

    recs.sort(key=lambda x: -abs(x.get("penalty") or 0))
    return recs[:max_n]


if __name__ == "__main__":
    # demo
    triggered = [
        {"code": "MessageDescrMinLen", "category": "Description", "penalty": 5},
        {"code": "MessageTxtfNoComposition", "category": "Characteristics", "penalty": 15},
        {"code": "MessageTooSmallItem", "category": "Image", "penalty": 1.5},
        {"code": "MessageTitleMarketingText", "category": "Title", "penalty": 5},
    ]
    state = {
        "name": "X",
        "description": "短",
        "photo_count": 1,
        "compositions": [],
        "n_options": 0,
    }
    for r in build_recommendations(triggered, state):
        print(f"  [{r['category']}] (-{r['penalty']}) {r['code']}")
        print(f"      → {r['fix']}")