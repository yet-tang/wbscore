"""fetch_wb_data.py — 从 WB 公开搜索 API 抓商品数据.

用法:
    pip3 install --user requests  (如果还没装)
    python3 fetch_wb_data.py --n 10000 --out data/wb_real_products.jsonl

API 端点: https://search.wb.ru/exactmatch/ru/common/v8/search
无需 API key,无需登录,无认证。
每页最多 100 商品,但 WB 单关键词最多返回 3 页(~300 条)。
所以 10000 条需要遍历多个关键词/类目。
"""
from __future__ import annotations
import json, time, argparse, random, sys
from pathlib import Path
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://search.wb.ru/exactmatch/ru/common/v8/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Accept": "application/json",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "x-client-name": "site",
    "x-client-version": "1.0.0",
}

# 覆盖俄语 WB 主要类目的关键词列表(每个 ~300 商品)
KEYWORDS = [
    # 服装
    "платье женское", "юбка женская", "блузка женская", "рубашка мужская",
    "джинсы мужские", "джинсы женские", "куртка мужская", "куртка женская",
    "пальто женское", "свитер женский", "толстовка мужская", "худи мужское",
    "спортивный костюм женский", "спортивный костюм мужской",
    "белье женское", "бюстгальтер", "купальник женский", "носки мужские",
    "колготки женские", "шорты мужские", "шорты женские", "брюки женские",
    "брюки мужские", "водолазка женская", "пиджак мужской", "туфли женские",
    # 鞋
    "кроссовки женские", "кроссовки мужские", "ботинки мужские",
    "сапоги женские", "туфли мужские", "тапочки домашние",
    # 配件
    "сумка женская", "рюкзак", "кошелек женский", "ремень мужской",
    "шапка женская", "шарф женский", "перчатки кожаные",
    # 电子
    "наушники", "смартфон", "телефон", "планшет",
    "ноутбук", "монитор", "клавиатура", "мышь компьютерная",
    "колонка bluetooth", "power bank", "зарядное устройство",
    "телевизор", "камера видеонаблюдения", "флешка usb",
    # 家居
    "подушка", "одеяло", "постельное белье", "полотенце",
    "сковорода", "кастрюля", "чайник электрический", "кружка керамическая",
    "тарелка", "контейнер для еды", "мусорное ведро",
    # 美妆
    "шампунь", "крем для лица", "помада", "тушь для ресниц",
    "парфюм женский", "парфюм мужской", "духи", "гель для душа",
    # 母婴
    "подгузники", "детское питание", "коляска", "автокресло",
    "игрушка детская", "конструктор лего",
    # 运动
    "велосипед", "самокат", "ролики", "лыжи",
    "гантели", "коврик для йоги", "мяч футбольный",
    # 玩具/爱好
    "настольная игра", "пазл", "конструктор",
    # 工具/家居改善
    "дрель", "шуруповерт", "набор инструментов", "лампа светодиодная",
    # 汽车
    "автомобильный держатель", "видеорегистратор",
    # 宠物
    "корм для собак", "корм для кошек",
    # 图书/办公
    "книга", "ежедневник", "ручка шариковая",
]


def make_session():
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def search_products(session, query: str, page: int = 1) -> Dict:
    """调 WB 公开搜索 API 返回 JSON."""
    params = {
        "ab_testing": "false",
        "appType": "1",
        "curr": "rub",
        "dest": "-1257786",
        "inheritFilters": "false",
        "lang": "ru",
        "page": str(page),
        "query": query,
        "resultset": "catalog",
        "sort": "popular",
        "spp": "30",
        "suppressSpellcheck": "false",
    }
    r = session.get(API_BASE, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_products(payload: dict) -> List[Dict]:
    """从 v8 搜索响应里提取商品核心字段。

    v8 字段结构:
      sizes[i].price: {basic: kopecks, product: kopecks, logistics, return, cashback, wallet}
      pics: [str, ...] 主图 URL 列表(basket 域名)
    """
    out = []
    for p in payload.get("products") or []:
        try:
            sizes = p.get("sizes") or []
            price_basic_rub = price_sale_rub = None
            if sizes and isinstance(sizes[0], dict):
                price = sizes[0].get("price")
                if isinstance(price, dict):
                    if "basic" in price and isinstance(price["basic"], (int, float)):
                        price_basic_rub = int(price["basic"]) // 100
                    if "product" in price and isinstance(price["product"], (int, float)):
                        price_sale_rub = int(price["product"]) // 100
            pics_count = p.get("pics")
            # WB v8: 'pics' 字段是 int(图片张数),不是 URL 列表
            # 主图 URL 按 id 计算 vol/part 路径,用 images/big/1.webp
            if isinstance(pics_count, int) and pics_count > 0:
                pid = p.get("id") or 0
                main_image = (
                    f"https://basket-01.wbbasket.ru/vol{(pid // 100000) % 100}"
                    f"/part{(pid // 1000) % 100}/{pid}/images/big/1.webp"
                )
            else:
                main_image = None
        except Exception as e:
            print(f"  skip product id={p.get('id')}: {e}", file=__import__('sys').stderr)
            continue

        out.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "brand": p.get("brand"),
            "brandId": p.get("brandId"),
            "supplier": p.get("supplier"),
            "supplierId": p.get("supplierId"),
            "supplierRating": p.get("supplierRating"),
            "price_basic_rub": price_basic_rub,
            "price_sale_rub": price_sale_rub,
            "discount_pct": (
                round(100 * (1 - price_sale_rub / price_basic_rub))
                if price_basic_rub and price_sale_rub and price_basic_rub > 0
                else None
            ),
            "rating": p.get("rating"),
            "reviewRating": p.get("reviewRating"),
            "feedbacks": p.get("feedbacks"),
            "totalQuantity": p.get("totalQuantity"),
            "subjectId": p.get("subjectId"),
            "subjectParentId": p.get("subjectParentId"),
            "n_sizes": len(sizes),
            "main_image": main_image,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="目标商品数")
    ap.add_argument("--keywords", type=int, default=0, help="使用前 N 个关键词(0=全部)")
    ap.add_argument("--per_keyword", type=int, default=3, help="每个关键词最多翻 N 页")
    ap.add_argument("--out", default="data/wb_real_products.jsonl")
    ap.add_argument("--sleep", type=float, default=0.4, help="每个请求间隔(秒)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    keywords = KEYWORDS if args.keywords == 0 else KEYWORDS[:args.keywords]
    print(f"Will fetch up to {args.n} products across {len(keywords)} keywords")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = make_session()

    seen_ids = set()
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for kw in keywords:
            if n_written >= args.n:
                break
            for page in range(1, args.per_keyword + 1):
                if n_written >= args.n:
                    break
                try:
                    payload = search_products(session, kw, page)
                    prods = extract_products(payload)
                    if not prods:
                        break
                    for p in prods:
                        if p["id"] in seen_ids:
                            continue
                        seen_ids.add(p["id"])
                        p["search_keyword"] = kw
                        p["search_page"] = page
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                        n_written += 1
                        if n_written >= args.n:
                            break
                    time.sleep(args.sleep + random.random() * 0.2)
                except Exception as e:
                    print(f"  Error on '{kw}' page {page}: {e}", file=sys.stderr)
                    time.sleep(2)
                    continue
            print(f"  '{kw}': {n_written} total so far")

    print(f"\nDone. Wrote {n_written} unique products to {out_path}")


if __name__ == "__main__":
    main()