#!/usr/bin/env python3
"""
Generate synthetic WB products with DELIBERATE rule violations.

Goal: fix the class imbalance in training data. Real training data is mostly
high-quality products, so FAIL signal is rare for most rules. Model never
learns what bad-text looks like → at inference time it confidently invents
violations.

Each generated product targets 1-4 specific rules to violate, so every rule
ends up with 200-500 negative training samples.
"""
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

WORKSPACE = Path(__file__).parent
OUT_FILE = WORKSPACE / "data" / "synthetic_negatives.parquet"

random.seed(42)


def make(name, description, options, comps, photos, has_video, brand):
    return {
        "nm_id": random.randint(2_000_000_000, 9_999_999_999),
        "name": name,
        "description": description,
        "options_json": json.dumps(options, ensure_ascii=False),
        "compositions_json": json.dumps(comps, ensure_ascii=False),
        "photo_count": photos,
        "has_video": has_video,
        "brand": brand,
        "imt_id": random.randint(100_000_000, 999_999_999),
        "subj_name": "Тестовая категория",
        "subj_root_name": "Одежда",
        "price_sale": random.randint(1000, 5000),
        "price_basic": random.randint(5000, 10000),
        "rating": round(random.uniform(0, 5), 1),
        "feedbacks": random.randint(0, 100),
        "description_len": len(description),
        "n_options": len(options),
        "is_apparel": True,
    }


# === Synthetic negative samples: each one targets specific rule violations ===

def gen_obscene():
    """Contains real Russian obscenities in title or description."""
    obs = ["хуй", "пизда", "блядь", "сука", "жопа", "говно", "ебать"]
    word = random.choice(obs)
    return make(
        name=f"Качественный товар с {word} в комплекте",
        description=f"Отличный товар. Не {word}, а находка! Описание с подробностями. " * 5,
        options=[{"name": "Состав", "value": "хлопок 95%"}, {"name": "Цвет", "value": "красный"}, {"name": "Размер", "value": "42"}],
        comps=["хлопок 95%"],
        photos=8, has_video=True, brand="BrandX"
    )


def gen_word_duplicates():
    """Title has the same word 3+ times."""
    word = random.choice(["платье", "костюм", "кроссовки", "туфли", "сумка"])
    return make(
        name=f"{word} {word} {word} {word} женский стильный",
        description="Хорошее описание товара с подробностями. " * 8,
        options=[{"name": "Состав", "value": "хлопок"}, {"name": "Цвет", "value": "красный"}, {"name": "Размер", "value": "42"}],
        comps=["хлопок"],
        photos=6, has_video=False, brand="Brand"
    )


def gen_marketing():
    """Title has 'лучший', 'топ', 'хит', 'акция' etc."""
    m = random.choice(["лучший", "топ", "хит продаж", "акция", "premium", "эксклюзив"])
    return make(
        name=f"{m} платье женское Elegant Style новинка",
        description="Подробное описание товара. " * 10,
        options=[{"name": "Состав", "value": "хлопок"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_no_title():
    """Title is too short."""
    return make(
        name="X",
        description="Подробное описание товара с подробностями. " * 10,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=6, has_video=False, brand="Brand"
    )


def gen_title_too_long():
    """Title exceeds 60 chars."""
    return make(
        name="Очень длинное название товара которое превышает шестьдесят символов и содержит много лишних слов",
        description="Нормальное описание товара. " * 8,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_title_all_caps():
    """Title is mostly uppercase English."""
    return make(
        name="PREMIUM QUALITY BRAND NEW STYLE FASHION ITEM",
        description="Normal product description. " * 8,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=6, has_video=False, brand="Brand"
    )


def gen_no_composition():
    """Clothing item without Состав field."""
    return make(
        name="Платье летнее женское Elegant Style",
        description="Подробное описание товара. " * 10,
        options=[{"name": "Цвет", "value": "красный"}, {"name": "Размер", "value": "42"}],  # no Состав
        comps=[],  # no composition
        photos=8, has_video=True, brand="Brand"
    )


def gen_missing_charcs():
    """Too few characteristics."""
    return make(
        name="Товар хороший брендовый",
        description="Описание товара. " * 10,
        options=[],  # no characteristics at all
        comps=["хлопок"],
        photos=5, has_video=False, brand="Brand"
    )


def gen_short_description():
    """Description < 200 chars."""
    return make(
        name="Платье летнее женское стильное",
        description="Краткое описание.",  # 19 chars
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_too_few_photos():
    """Less than 4 photos."""
    n = random.randint(1, 3)
    return make(
        name="Платье летнее женское стильное",
        description="Подробное описание товара с подробностями. " * 8,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=n, has_video=False, brand="Brand"
    )


def gen_another_brand():
    """Description mentions another brand name."""
    other_brand = random.choice(["Nike", "Zara", "Adidas", "Puma", "Gucci", "Prada", "Apple", "Samsung"])
    return make(
        name="Платье летнее женское стильное",
        description=f"Качественный аналог {other_brand}, такой же стиль. " * 10,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="MyBrand"
    )


def gen_description_with_link():
    """Description contains URL."""
    return make(
        name="Платье летнее женское стильное",
        description="Смотрите подробности на нашем сайте https://example.com/promo и в инсте @brand. " * 5,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_description_with_phone():
    """Description contains phone number."""
    return make(
        name="Платье летнее женское стильное",
        description="Звоните +7 999 123 45 67 для заказа. " * 10,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_description_nmid():
    """Description has SKU-like number string."""
    return make(
        name="Платье летнее женское стильное",
        description="Артикул 1234567. Описание товара с подробностями. " * 10,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


def gen_title_nmid():
    """Title has SKU-like digits."""
    return make(
        name="Платье летнее 1234567 женское стильное",
        description="Описание товара с подробностями. " * 10,
        options=[{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
        comps=["хлопок"],
        photos=8, has_video=True, brand="Brand"
    )


# Distribution: each rule should get ~150 fail samples
GENERATORS = [
    (gen_obscene, 200),                # 0% -> ~2% fail rate for obscene
    (gen_word_duplicates, 200),        # 1.9% -> ~4%
    (gen_marketing, 200),
    (gen_no_title, 150),
    (gen_title_too_long, 150),
    (gen_title_all_caps, 150),
    (gen_no_composition, 200),
    (gen_missing_charcs, 200),
    (gen_short_description, 200),
    (gen_too_few_photos, 200),
    (gen_another_brand, 250),          # 0% -> ~2-3%
    (gen_description_with_link, 150),
    (gen_description_with_phone, 150),
    (gen_description_nmid, 150),
    (gen_title_nmid, 150),
]


def main():
    rows = []
    print(f"Generating {sum(n for _, n in GENERATORS)} synthetic negative samples...")
    for gen, n in GENERATORS:
        for _ in range(n):
            rows.append(gen())
    random.shuffle(rows)
    print(f"Generated {len(rows)} products")

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT_FILE)
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()