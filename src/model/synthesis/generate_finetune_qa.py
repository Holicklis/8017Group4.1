from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("•", " ")
    return text


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _is_noisy_sentence(text: str) -> bool:
    t = _normalize_text(text).lower()
    if len(t) < 25:
        return True
    noise_patterns = [
        r"all rights reserved",
        r"take no responsibility",
        r"no representation as to",
        r"copyright",
        r"http://",
        r"https://",
        r"\b\d{6,}\b",  # long document ids / codes
        r"forms an integral part of",
        r"shall be deemed to be deleted and replaced",
        r"under the section entitled",
        r"with the prior approval of the promoter",
    ]
    return any(re.search(p, t) for p in noise_patterns)


def _compact_answer(text: str, max_chars: int) -> str:
    text = _normalize_text(text)
    # Remove dangling punctuation spacing artifacts.
    text = text.replace(" ,", ",").replace(" .", ".")
    return text[:max_chars].strip()


def _is_bad_qa_answer(answer: str, source_tag: str) -> bool:
    a = _normalize_text(answer).lower()
    if len(a) < 35:
        return True

    bad_patterns = [
        r"forms an integral part of",
        r"shall be deemed to be deleted and replaced",
        r"under the section entitled",
        r"with the prior approval of the promoter",
        r"this \w+ addendum",
        r"dated \d{1,2} [a-z]+ \d{4}",
    ]
    if any(re.search(p, a) for p in bad_patterns):
        return True

    # Fee-tagged answers should include fee-like content, not generic legal text.
    if "fees_charges" in str(source_tag).lower():
        if not re.search(r"\b(fee|fees|charge|charges|expense|ongoing)\b|費用|收費|管理費|開支", a):
            return True

    return False


def _normalized_question_key(question: str) -> str:
    q = _normalize_text(question).lower()
    q = q.replace("？", "?")
    q = re.sub(r"\s+", " ", q)
    return q


def _is_question_like(question: str, language: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return False

    if language.startswith("zh"):
        # Avoid generic statements and enforce interrogative markers.
        bad_prefix = ["請根據基金文件說明", "請總結", "請介紹"]
        if any(q.startswith(p) for p in bad_prefix):
            return False
        return ("？" in q) or any(
            k in q
            for k in [
                "是否",
                "甚麼",
                "什麼",
                "如何",
                "哪些",
                "哪個",
                "何時",
                "多少",
                "風險",
            ]
        )

    # English: must look like a question.
    wh_tokens = (
        "what",
        "who",
        "how",
        "when",
        "why",
        "which",
        "does",
        "is",
        "are",
        "under what",
    )
    ql = q.lower()
    return q.endswith("?") and ql.startswith(wh_tokens)


def _infer_ticker_from_path(csv_dir: Path, default_ticker: str = "UNKNOWN") -> str:
    # Expected structure: .../documentation/<ticker>/csv
    parent = csv_dir.parent
    if parent.name.lower() == "csv":
        parent = parent.parent
    name = parent.name.strip()
    if re.fullmatch(r"\d{4,5}", name):
        return name.zfill(5)
    return default_ticker


@dataclass
class QAPair:
    ticker: str
    question: str
    answer: str
    source_file: str
    source_sentence_id: Optional[int] = None
    source_tag: str = "heuristic"
    language: str = "en"

    def to_dict(self, idx: int) -> Dict[str, object]:
        return {
            "id": f"{self.ticker}_qa_{idx:05d}",
            "ticker": self.ticker,
            "language": self.language,
            "question": self.question,
            "answer": self.answer,
            "source_file": self.source_file,
            "source_sentence_id": self.source_sentence_id,
            "source_tag": self.source_tag,
        }


class ETFQAGenerator:
    def __init__(
        self,
        ticker: str,
        min_answer_chars: int = 35,
        max_answer_chars: int = 520,
        seed: int = 42,
        only_key_facts: bool = False,
    ) -> None:
        self.ticker = ticker
        self.min_answer_chars = min_answer_chars
        self.max_answer_chars = max_answer_chars
        self.rng = random.Random(seed)
        self.only_key_facts = only_key_facts

    @staticmethod
    def _is_key_facts_file(path: Path) -> bool:
        name = path.name.lower()
        key_tokens = [
            "key_facts",
            "keyfacts",
            "product_key_facts",
            "product key facts",
            "kfs",
        ]
        return any(token in name for token in key_tokens)

    def _load_sentences(self, csv_dir: Path) -> pd.DataFrame:
        files = sorted(csv_dir.glob("*.csv"))
        if self.only_key_facts:
            files = [f for f in files if self._is_key_facts_file(f)]
        if not files:
            if self.only_key_facts:
                raise FileNotFoundError(f"No Key Facts CSV files found in {csv_dir}")
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        frames: List[pd.DataFrame] = []
        for f in files:
            df = pd.read_csv(f)
            if "text" not in df.columns:
                continue
            local = df.copy()
            if "sentence_id" not in local.columns:
                local["sentence_id"] = range(len(local))
            local["text"] = local["text"].fillna("").astype(str).map(_normalize_text)
            local["source_file"] = f.name
            local = local[~local["text"].map(_is_noisy_sentence)]
            frames.append(local[["sentence_id", "text", "source_file"]])

        if not frames:
            raise ValueError(f"No usable text rows found in CSV files under {csv_dir}")

        merged = pd.concat(frames, ignore_index=True)
        merged = merged[merged["text"].str.len() >= self.min_answer_chars]
        merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
        return merged

    def _extract_key_facts(self, corpus_text: str) -> Dict[str, str]:
        patterns = {
            "stock_code": r"Stock code:\s*([^\.]+?)(?:\s{2,}|Trading lot size:|Fund manager:|$)",
            "trading_lot_size": r"Trading lot size:\s*([^\.]+?)(?:\s{2,}|Fund manager:|Underlying index:|$)",
            "fund_manager": r"Fund manager:\s*([^\.]+?)(?:\s{2,}|Trustee and Custodian:|Underlying index:|$)",
            "underlying_index": r"Underlying index:\s*([^\.]+?)(?:\s{2,}|Trading currency:|Base currency:|$)",
            "base_currency": r"Base currency:\s*([^\.]+?)(?:\s{2,}|Ongoing charges|Financial year end|$)",
            "ongoing_charges": r"Ongoing charges over a year[#:\s]*([^\.]+?)(?:\s{2,}|Tracking difference|$)",
            "tracking_difference": r"Tracking difference[^:]*:\s*([^\.]+?)(?:\s{2,}|Financial year end|$)",
            "dividend_policy": r"Dividend policy[#:\s]*([^\.]+?)(?:\s{2,}|website|$)",
        }

        out: Dict[str, str] = {}
        for key, pat in patterns.items():
            m = re.search(pat, corpus_text, flags=re.IGNORECASE)
            if m:
                out[key] = _normalize_text(m.group(1))
        return out

    def _pick_rows(self, df: pd.DataFrame, pattern: str, limit: int = 5) -> List[Dict[str, object]]:
        mask = df["text"].str.contains(pattern, case=False, regex=True, na=False)
        sub = df[mask].copy().head(limit)
        return sub.to_dict(orient="records")

    def _qa_from_facts(self, facts: Dict[str, str]) -> List[QAPair]:
        qa: List[QAPair] = []
        mapping = [
            (
                "stock_code",
                "What is the stock code of ETF {ticker}?",
                "The stock code is {value}.",
            ),
            (
                "trading_lot_size",
                "What is the trading lot size for ETF {ticker}?",
                "The trading lot size is {value}.",
            ),
            (
                "fund_manager",
                "Who is the fund manager of ETF {ticker}?",
                "The fund manager is {value}.",
            ),
            (
                "underlying_index",
                "What index does ETF {ticker} track?",
                "The ETF tracks {value}.",
            ),
            (
                "base_currency",
                "What is the base currency of ETF {ticker}?",
                "The base currency is {value}.",
            ),
            (
                "ongoing_charges",
                "What are the ongoing charges of ETF {ticker}?",
                "The ongoing charges are {value}.",
            ),
            (
                "tracking_difference",
                "What is the latest tracking difference for ETF {ticker}?",
                "The latest tracking difference is {value}.",
            ),
            (
                "dividend_policy",
                "What is the dividend policy of ETF {ticker}?",
                "The dividend policy is {value}.",
            ),
        ]
        for key, q_tpl, a_tpl in mapping:
            value = facts.get(key, "").strip()
            if not value:
                continue
            qa.append(
                QAPair(
                    ticker=self.ticker,
                    question=q_tpl.format(ticker=self.ticker),
                    answer=a_tpl.format(value=value),
                    source_file="multi_csv_aggregate",
                    source_tag="fact_extraction",
                    language="en",
                )
            )
        return qa

    def _qa_from_topics(self, df: pd.DataFrame) -> List[QAPair]:
        qa: List[QAPair] = []

        topic_specs = [
            {
                "tag": "objective_strategy",
                "pattern": r"\bobjective\b|\binvestment objective\b|\bstrategy\b|\brebalance\b",
                "questions": [
                    "What is the investment objective of ETF {ticker}?",
                    "How does ETF {ticker} implement its index-tracking strategy?",
                    "How closely does ETF {ticker} aim to replicate its benchmark?",
                ],
            },
            {
                "tag": "key_risks",
                "pattern": r"\bkey risks\b|\binvestment risk\b|\bindex risk\b|\btracking error\b|\bconcentration\b|\bvolatil",
                "questions": [
                    "What are the key risks investors should know for ETF {ticker}?",
                    "What risk factors are highlighted in the product key facts for ETF {ticker}?",
                    "Which risks could cause losses for ETF {ticker} investors?",
                ],
            },
            {
                "tag": "fees_charges",
                "pattern": r"\bfees\b|\bcharges\b|\bmanagement fee\b|\btrustee fee\b|\bstamp duty\b|\btrading fee\b",
                "questions": [
                    "What fees and charges apply to ETF {ticker}?",
                    "What are the key trading and ongoing costs for ETF {ticker}?",
                    "How do management and trustee fees work for ETF {ticker}?",
                ],
            },
            {
                "tag": "derivatives",
                "pattern": r"\bderivative\b|\bfutures\b|\boptions\b|\bwarrants\b|\bnet derivative exposure\b",
                "questions": [
                    "Does ETF {ticker} use derivatives, and what are the limits?",
                    "What is the derivative exposure framework for ETF {ticker}?",
                    "How are futures/options/warrants addressed in ETF {ticker} documentation?",
                ],
            },
            {
                "tag": "termination",
                "pattern": r"\btermination\b|\bsuccessor index\b|\brolling three month\b|\bhk\\$3 billion\b",
                "questions": [
                    "What are the termination conditions for ETF {ticker}?",
                    "Under what scenarios may ETF {ticker} be terminated?",
                    "How does low NAV affect termination risk for ETF {ticker}?",
                ],
            },
            {
                "tag": "dividend",
                "pattern": r"\bdividend\b|\bsemi-annually\b|\bdistribution\b",
                "questions": [
                    "How does dividend distribution work for ETF {ticker}?",
                    "When are dividends generally paid for ETF {ticker}?",
                    "Are dividend payouts guaranteed for ETF {ticker}?",
                ],
            },
            {
                "tag": "currency_counter",
                "pattern": r"\bhkd counter\b|\brmb counter\b|\bdual counter\b|\btrading currency\b",
                "questions": [
                    "How do HKD and RMB counters work for ETF {ticker}?",
                    "What dual-counter risks should investors know for ETF {ticker}?",
                    "What currency-related trading considerations apply to ETF {ticker}?",
                ],
            },
        ]

        for spec in topic_specs:
            rows = self._pick_rows(df, pattern=spec["pattern"], limit=8)
            if not rows:
                continue

            # Build answer chunks and generate multiple QAs per topic.
            base_sentences = [_normalize_text(str(r["text"])) for r in rows]
            clean_sentences = [s for s in base_sentences if len(s) >= self.min_answer_chars]
            if not clean_sentences:
                continue

            for q_tpl in spec["questions"]:
                # Mix 2-4 sentences to increase diversity but keep grounded.
                take_n = min(len(clean_sentences), self.rng.randint(2, 4))
                picked = self.rng.sample(clean_sentences, k=take_n)
                answer = _compact_answer(" ".join(picked), max_chars=self.max_answer_chars)
                if len(answer) < self.min_answer_chars:
                    continue
                qa.append(
                    QAPair(
                        ticker=self.ticker,
                        question=q_tpl.format(ticker=self.ticker),
                        answer=answer,
                        source_file=str(rows[0].get("source_file", "")),
                        source_sentence_id=int(rows[0].get("sentence_id", 0)),
                        source_tag=spec["tag"],
                        language="en",
                    )
                )
        return qa

    def _qa_from_single_sentence(self, df: pd.DataFrame) -> List[QAPair]:
        qa: List[QAPair] = []
        templates = [
            (
                r"\bpassive exchange traded fund\b|\bpassive investment\b",
                "Is ETF {ticker} actively managed or passive?",
            ),
            (
                r"\bno guarantee\b|\brepayment of principal\b",
                "Is principal guaranteed when investing in ETF {ticker}?",
            ),
            (r"\btracking error\b", "What causes tracking error in ETF {ticker}?"),
            (
                r"\bequity market risk\b|\bequity securities\b",
                "What equity market risks affect ETF {ticker}?",
            ),
        ]
        for pattern, q_tpl in templates:
            rows = self._pick_rows(df, pattern=pattern, limit=3)
            for row in rows:
                answer = _compact_answer(str(row["text"]), max_chars=self.max_answer_chars)
                if len(answer) < self.min_answer_chars:
                    continue
                qa.append(
                    QAPair(
                        ticker=self.ticker,
                        question=q_tpl.format(ticker=self.ticker),
                        answer=answer,
                        source_file=str(row.get("source_file", "")),
                        source_sentence_id=int(row.get("sentence_id", 0)),
                        source_tag="single_sentence",
                        language="en",
                    )
                )
        return qa

    def _to_zh_hk(self, qa: QAPair) -> QAPair:
        # Template-level bilingual expansion (deterministic, no external API).
        q = qa.question
        replacements = [
            (f"ETF {self.ticker}", f"{self.ticker} ETF"),
            ("What is the stock code of", "請問"),
        ]
        _ = replacements  # keep lints happy if not used in some branches

        zh_q_map = {
            "What is the stock code": "這隻ETF的股票代號是甚麼",
            "What is the trading lot size": "這隻ETF每手交易單位是多少",
            "Who is the fund manager": "這隻ETF的基金經理是誰",
            "What index does": "這隻ETF追蹤哪個指數",
            "What is the base currency": "這隻ETF的基礎貨幣是甚麼",
            "What are the ongoing charges": "這隻ETF的全年持續費用是多少",
            "What is the latest tracking difference": "這隻ETF最新追蹤偏離是多少",
            "What is the dividend policy": "這隻ETF的派息政策是甚麼",
            "What is the investment objective": "這隻ETF的投資目標是甚麼",
            "How does": "這隻ETF如何執行其追蹤策略",
            "What are the key risks": "這隻ETF有哪些主要風險",
            "What risk factors are highlighted": "這隻ETF的產品資料概要重點風險有哪些",
            "Which risks could cause losses": "哪些風險可能導致這隻ETF出現虧損",
            "What fees and charges": "這隻ETF涉及哪些費用與收費",
            "What are the key trading and ongoing costs": "這隻ETF的交易與持有成本有哪些",
            "Does": "這隻ETF是否使用衍生工具，以及有甚麼限制",
            "How are futures/options/warrants addressed": "這隻ETF在文件中如何規範期貨、期權及認股權證",
            "What are the termination conditions": "這隻ETF在甚麼情況下可能終止",
            "Under what scenarios may": "這隻ETF在甚麼情況下可能被終止",
            "How does low NAV affect termination risk": "資產淨值偏低如何影響這隻ETF的終止風險",
            "How does dividend distribution work": "這隻ETF如何派發股息",
            "When are dividends generally paid": "這隻ETF一般在何時派息",
            "Are dividend payouts guaranteed": "這隻ETF派息是否有保證",
            "How do HKD and RMB counters work": "這隻ETF的港幣／人民幣櫃台如何運作",
            "What dual-counter risks": "這隻ETF的雙櫃台交易有甚麼風險",
            "What currency-related trading considerations": "這隻ETF在貨幣及交易櫃台方面有甚麼注意事項",
            "Is": "這隻ETF是否屬於被動管理",
            "What causes tracking error": "這隻ETF的追蹤誤差主要由甚麼造成",
            "What equity market risks affect": "這隻ETF面對哪些股票市場風險",
        }

        zh_question = ""
        for en_prefix, zh_prefix in zh_q_map.items():
            if q.startswith(en_prefix):
                zh_question = f"{zh_prefix}（{self.ticker}）？"
                break
        if not zh_question:
            return None

        zh_answer = f"根據基金文件：{qa.answer}"
        return QAPair(
            ticker=qa.ticker,
            question=zh_question,
            answer=zh_answer,
            source_file=qa.source_file,
            source_sentence_id=qa.source_sentence_id,
            source_tag=f"{qa.source_tag}_zh",
            language="zh-HK",
        )

    def build_qa_pairs(self, csv_dir: Path, max_pairs: int = 240, include_zh: bool = True) -> List[QAPair]:
        df = self._load_sentences(csv_dir)
        corpus_text = " ".join(df["text"].tolist())
        facts = self._extract_key_facts(corpus_text)

        qa: List[QAPair] = []
        qa.extend(self._qa_from_facts(facts=facts))
        qa.extend(self._qa_from_topics(df))
        qa.extend(self._qa_from_single_sentence(df))

        # Deduplicate by normalized question-answer pair.
        seen = set()
        deduped: List[QAPair] = []
        for item in qa:
            key = (
                _normalize_text(item.question).lower(),
                _normalize_text(item.answer).lower(),
                item.language,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if include_zh:
            zh_pairs = [self._to_zh_hk(x) for x in deduped]
            for item in zh_pairs:
                if item is None:
                    continue
                key = (
                    _normalize_text(item.question).lower(),
                    _normalize_text(item.answer).lower(),
                    item.language,
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

        # Quality gate: keep only real questions.
        deduped = [x for x in deduped if _is_question_like(x.question, x.language)]
        # Quality gate: remove legal boilerplate / low-value answers.
        deduped = [x for x in deduped if not _is_bad_qa_answer(x.answer, x.source_tag)]

        # Strong dedupe by normalized question text to avoid repeated prompts.
        # Keep the shortest non-trivial answer version per question.
        by_question: Dict[tuple[str, str], QAPair] = {}
        for item in deduped:
            q_key = (item.language, _normalized_question_key(item.question))
            old = by_question.get(q_key)
            if old is None:
                by_question[q_key] = item
                continue
            old_len = len(_normalize_text(old.answer))
            new_len = len(_normalize_text(item.answer))
            if new_len < old_len and new_len >= self.min_answer_chars:
                by_question[q_key] = item
        deduped = list(by_question.values())

        # Stable shuffle for variety then cap.
        self.rng.shuffle(deduped)
        return deduped[:max_pairs]


def _to_chatml(qa: Dict[str, object], system_prompt: str) -> Dict[str, object]:
    return {
        "id": qa["id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ],
        "metadata": {
            "ticker": qa["ticker"],
            "language": qa["language"],
            "source_file": qa["source_file"],
            "source_sentence_id": qa["source_sentence_id"],
            "source_tag": qa["source_tag"],
        },
    }


def generate_finetune_qa(
    csv_dir: Path,
    output_dir: Path,
    ticker: str = "02800",
    max_pairs: int = 240,
    include_zh: bool = True,
    only_key_facts: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = ETFQAGenerator(ticker=ticker, only_key_facts=only_key_facts)
    pairs = gen.build_qa_pairs(csv_dir=csv_dir, max_pairs=max_pairs, include_zh=include_zh)

    rows = [p.to_dict(idx=i + 1) for i, p in enumerate(pairs)]
    qa_df = pd.DataFrame(rows)

    stem = _safe_filename(f"{ticker}_finetune_qa")
    csv_path = output_dir / f"{stem}.csv"
    jsonl_path = output_dir / f"{stem}.jsonl"
    chatml_path = output_dir / f"{stem}_chatml.jsonl"
    summary_path = output_dir / f"{stem}_summary.json"

    qa_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    system_prompt = (
        "You are a professional Hong Kong ETF assistant. "
        "Answer based only on ETF documentation facts, clearly and conservatively."
    )
    with chatml_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_chatml(row, system_prompt=system_prompt), ensure_ascii=False) + "\n")

    summary = {
        "ticker": ticker,
        "input_dir": str(csv_dir),
        "num_csv_files": len(list(csv_dir.glob("*.csv"))),
        "num_qa_pairs": len(rows),
        "num_en_pairs": int((qa_df["language"] == "en").sum()) if not qa_df.empty else 0,
        "num_zh_pairs": int((qa_df["language"] != "en").sum()) if not qa_df.empty else 0,
        "max_pairs_requested": max_pairs,
        "include_zh": include_zh,
        "only_key_facts": only_key_facts,
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "chatml_jsonl": str(chatml_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def generate_finetune_qa_all(
    documentation_root: Path,
    output_dir: Path,
    max_pairs: int = 240,
    include_zh: bool = True,
    only_key_facts: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dirs = sorted({p.parent for p in documentation_root.glob("**/*.csv") if p.parent.name.lower() == "csv"})
    if not csv_dirs:
        raise FileNotFoundError(f"No csv directories found under: {documentation_root}")

    all_rows: List[Dict[str, object]] = []
    per_ticker: List[Dict[str, object]] = []

    for csv_dir in csv_dirs:
        ticker = _infer_ticker_from_path(csv_dir, default_ticker="UNKNOWN")
        sub_out = output_dir / "per_ticker"
        sub_out.mkdir(parents=True, exist_ok=True)

        try:
            result = generate_finetune_qa(
                csv_dir=csv_dir,
                output_dir=sub_out,
                ticker=ticker,
                max_pairs=max_pairs,
                include_zh=include_zh,
                only_key_facts=only_key_facts,
            )
            per_ticker.append(result)

            # Append generated rows to global aggregate.
            ticker_csv = Path(result["outputs"]["csv"])
            ticker_df = pd.read_csv(ticker_csv)
            if not ticker_df.empty:
                all_rows.extend(ticker_df.to_dict(orient="records"))
        except Exception as exc:
            per_ticker.append(
                {
                    "ticker": ticker,
                    "input_dir": str(csv_dir),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    combined_df = pd.DataFrame(all_rows)
    if not combined_df.empty:
        # Global dedupe by language + question.
        combined_df["q_key"] = (
            combined_df["language"].astype(str).str.lower()
            + "||"
            + combined_df["question"].astype(str).map(_normalized_question_key)
        )
        combined_df = combined_df.sort_values(["ticker", "source_tag"])
        combined_df = combined_df.drop_duplicates(subset=["q_key"], keep="first").drop(columns=["q_key"])
        combined_df = combined_df.reset_index(drop=True)
        combined_df["id"] = [f"global_qa_{i + 1:06d}" for i in range(len(combined_df))]

    combined_stem = "all_tickers_finetune_qa"
    combined_csv = output_dir / f"{combined_stem}.csv"
    combined_jsonl = output_dir / f"{combined_stem}.jsonl"
    combined_chatml = output_dir / f"{combined_stem}_chatml.jsonl"
    combined_summary = output_dir / f"{combined_stem}_summary.json"

    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    with combined_jsonl.open("w", encoding="utf-8") as f:
        for row in combined_df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    system_prompt = (
        "You are a professional Hong Kong ETF assistant. "
        "Answer based only on ETF documentation facts, clearly and conservatively."
    )
    with combined_chatml.open("w", encoding="utf-8") as f:
        for row in combined_df.to_dict(orient="records"):
            f.write(json.dumps(_to_chatml(row, system_prompt=system_prompt), ensure_ascii=False) + "\n")

    ok_runs = [x for x in per_ticker if x.get("status", "ok") != "failed"]
    fail_runs = [x for x in per_ticker if x.get("status") == "failed"]
    summary = {
        "mode": "all_csv",
        "documentation_root": str(documentation_root),
        "num_csv_dirs_scanned": len(csv_dirs),
        "num_tickers_succeeded": len(ok_runs),
        "num_tickers_failed": len(fail_runs),
        "num_total_pairs": int(len(combined_df)),
        "num_en_pairs": int((combined_df["language"] == "en").sum()) if not combined_df.empty else 0,
        "num_zh_pairs": int((combined_df["language"] != "en").sum()) if not combined_df.empty else 0,
        "include_zh": include_zh,
        "only_key_facts": only_key_facts,
        "max_pairs_per_ticker": max_pairs,
        "outputs": {
            "csv": str(combined_csv),
            "jsonl": str(combined_jsonl),
            "chatml_jsonl": str(combined_chatml),
            "summary": str(combined_summary),
            "per_ticker_dir": str(output_dir / "per_ticker"),
        },
        "failed_tickers": fail_runs[:50],
    }
    combined_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate larger clean Q&A fine-tuning datasets from ETF documentation sentence CSVs."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Path to one ETF documentation CSV folder",
    )
    parser.add_argument(
        "--documentation-root",
        type=Path,
        default=None,
        help="Root directory containing many ticker folders (e.g. data/etf/documentation)",
    )
    parser.add_argument(
        "--all-csv",
        action="store_true",
        help="Batch mode: parse all CSV folders under --documentation-root",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for QA datasets")
    parser.add_argument("--ticker", type=str, default="02800")
    parser.add_argument("--max-pairs", type=int, default=240)
    parser.add_argument(
        "--only-key-facts",
        action="store_true",
        help="Use only Product Key Facts / KFS CSV files as source (skip prospectus/addendum CSVs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all_csv:
        if args.documentation_root is None:
            raise ValueError("--documentation-root is required when --all-csv is enabled.")
        result = generate_finetune_qa_all(
            documentation_root=args.documentation_root,
            output_dir=args.output_dir,
            max_pairs=max(args.max_pairs, 20),
            include_zh=True,
            only_key_facts=args.only_key_facts,
        )
    else:
        if args.csv_dir is None:
            raise ValueError("--csv-dir is required when not using --all-csv.")
        result = generate_finetune_qa(
            csv_dir=args.csv_dir,
            output_dir=args.output_dir,
            ticker=args.ticker,
            max_pairs=max(args.max_pairs, 20),
            include_zh=True,
            only_key_facts=args.only_key_facts,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
