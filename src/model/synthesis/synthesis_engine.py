from __future__ import annotations

import json
import re
import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_LOCAL_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_POLICY_VERSION = "info-first-v2"


def _normalize_ticker(ticker: object) -> str:
    text = str(ticker).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    digits = re.sub(r"\D", "", text)
    if digits:
        return str(int(digits)).zfill(4)
    return text.upper()


def _detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _is_risk_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "risk",
            "risks",
            "volatility",
            "drawdown",
            "downside",
            "風險",
            "波動",
            "回撤",
            "下行",
        ],
    )


def _is_recommendation_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "recommend",
            "suggest",
            "alternative",
            "which etf",
            "buy",
            "switch",
            "allocate",
            "rebalance",
            "推薦",
            "建議",
            "替代",
            "買",
            "換",
            "配置",
            "再平衡",
        ],
    )


def _is_fee_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "fee",
            "fees",
            "charge",
            "charges",
            "expense ratio",
            "ongoing charges",
            "management fee",
            "費用",
            "收費",
            "管理費",
            "開支比率",
        ],
    )


def _is_dividend_query(text: str) -> bool:
    return _contains_any(text, ["dividend", "distribution", "yield", "派息", "股息", "分派"])


def _is_tracking_query(text: str) -> bool:
    return _contains_any(text, ["track", "tracked", "index", "benchmark", "objective", "追蹤", "指數", "目標"])


def _is_news_discovery_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "news",
            "headline",
            "event",
            "story",
            "related ticker",
            "find ticker",
            "which ticker",
            "資訊",
            "新聞",
            "事件",
            "相關代號",
            "找代號",
        ],
    )


def _is_model1_related_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "related etf",
            "similar etf",
            "alternative etf",
            "profile similar",
            "cluster",
            "hidden twin",
            "home bias",
            "相似",
            "替代",
            "同類",
            "叢集",
            "聚類",
            "隱藏雙生",
            "本地偏誤",
        ],
    )


def _is_feature_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "feature",
            "features",
            "overview",
            "summary",
            "about this etf",
            "tell me about",
            "characteristic",
            "profile",
            "介紹",
            "概覽",
            "重點",
            "特點",
            "這隻etf",
        ],
    )


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", _normalize_for_match(text)))


def _latest_synapse_topk_csv(root: Path) -> Path:
    files = sorted(root.glob("news_events_run_*/news_event_topk_matches.csv"))
    if not files:
        raise FileNotFoundError(
            f"No Synapse top-k output found under: {root}. Please run src/model/synapse/run_news_events.py first."
        )
    return files[-1]


@dataclass
class SynthesisConfig:
    dna_cluster_parquet: Path
    dna_home_bias_parquet: Path
    dna_hidden_twin_parquet: Path
    synapse_topk_csv: Optional[Path]
    synapse_output_root: Path
    similarity_threshold: float = 0.70
    max_alerts: int = 8
    top_alt_k: int = 3
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_query_similarity: bool = False
    enable_response_cache: bool = True
    response_cache_path: Optional[Path] = None
    llm_timeout_seconds: int = 90
    qa_per_ticker_root: Optional[Path] = None
    enable_direct_qa_lookup: bool = True

    @staticmethod
    def default() -> "SynthesisConfig":
        root = _project_root()
        dna_root = root / "model_output" / "dna"
        synapse_root = root / "model_output" / "synpse"
        return SynthesisConfig(
            dna_cluster_parquet=dna_root / "cluster_views" / "cluster_perspectives.parquet",
            dna_home_bias_parquet=dna_root / "advisory" / "home_bias_candidates.parquet",
            dna_hidden_twin_parquet=dna_root / "advisory" / "hidden_twin_candidates.parquet",
            synapse_topk_csv=None,
            synapse_output_root=synapse_root,
            response_cache_path=root / "model_output" / "Synthesis" / "cache" / "synthesis_response_cache.json",
            qa_per_ticker_root=root / "model_output" / "Synthesis" / "finetune_all" / "per_ticker",
        )


class SynthesisEngine:
    """
    Final synthesis interface:
    1) Pull DNA cluster + alternatives
    2) Pull Synapse alerts (similarity threshold)
    3) Generate a multilingual advisor-style answer with local Qwen
    """

    def __init__(self, config: Optional[SynthesisConfig] = None) -> None:
        self.config = config or SynthesisConfig.default()
        self._cluster_df: Optional[pd.DataFrame] = None
        self._home_bias_df: Optional[pd.DataFrame] = None
        self._hidden_twin_df: Optional[pd.DataFrame] = None
        self._synapse_df: Optional[pd.DataFrame] = None
        self._sentence_encoder = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._qa_df_cache: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _read_parquet_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet support is required for synthesis DNA inputs. Install one engine, e.g. `pip install pyarrow`."
            ) from exc

    def _load_dna_cluster(self) -> pd.DataFrame:
        if self._cluster_df is None:
            self._cluster_df = self._read_parquet_safe(self.config.dna_cluster_parquet).copy()
            self._cluster_df["ticker"] = self._cluster_df["ticker"].map(_normalize_ticker)
        return self._cluster_df

    def _load_home_bias(self) -> pd.DataFrame:
        if self._home_bias_df is None:
            self._home_bias_df = self._read_parquet_safe(self.config.dna_home_bias_parquet).copy()
            if "source_ticker" in self._home_bias_df.columns:
                self._home_bias_df["source_ticker"] = self._home_bias_df["source_ticker"].map(_normalize_ticker)
            if "alternative_ticker" in self._home_bias_df.columns:
                self._home_bias_df["alternative_ticker"] = self._home_bias_df["alternative_ticker"].map(
                    _normalize_ticker
                )
        return self._home_bias_df

    def _load_hidden_twin(self) -> pd.DataFrame:
        if self._hidden_twin_df is None:
            self._hidden_twin_df = self._read_parquet_safe(self.config.dna_hidden_twin_parquet).copy()
            if "ticker_a" in self._hidden_twin_df.columns:
                self._hidden_twin_df["ticker_a"] = self._hidden_twin_df["ticker_a"].map(_normalize_ticker)
            if "ticker_b" in self._hidden_twin_df.columns:
                self._hidden_twin_df["ticker_b"] = self._hidden_twin_df["ticker_b"].map(_normalize_ticker)
        return self._hidden_twin_df

    def _load_synapse_topk(self) -> pd.DataFrame:
        if self._synapse_df is None:
            csv_path = self.config.synapse_topk_csv or _latest_synapse_topk_csv(self.config.synapse_output_root)
            df = pd.read_csv(csv_path).copy()
            if "predicted_ticker" not in df.columns:
                raise ValueError(f"Synapse file missing predicted_ticker: {csv_path}")
            df["predicted_ticker"] = df["predicted_ticker"].map(_normalize_ticker)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            self._synapse_df = df
        return self._synapse_df

    def _response_cache_key(
        self,
        ticker: str,
        user_query: str,
        backend: str,
        qwen_model: str,
    ) -> str:
        ticker_norm = _normalize_ticker(ticker)
        synapse_path = self.config.synapse_topk_csv or _latest_synapse_topk_csv(self.config.synapse_output_root)
        signatures = {
            "prompt_policy_version": PROMPT_POLICY_VERSION,
            "ticker": ticker_norm,
            "user_query": user_query.strip(),
            "backend": backend,
            "qwen_model": qwen_model,
            "similarity_threshold": self.config.similarity_threshold,
            "max_alerts": self.config.max_alerts,
            "top_alt_k": self.config.top_alt_k,
            "enable_query_similarity": self.config.enable_query_similarity,
            "enable_direct_qa_lookup": self.config.enable_direct_qa_lookup,
            "qa_per_ticker_root": str(self.config.qa_per_ticker_root) if self.config.qa_per_ticker_root else None,
            "dna_cluster_mtime": self.config.dna_cluster_parquet.stat().st_mtime
            if self.config.dna_cluster_parquet.exists()
            else None,
            "dna_home_bias_mtime": self.config.dna_home_bias_parquet.stat().st_mtime
            if self.config.dna_home_bias_parquet.exists()
            else None,
            "dna_hidden_twin_mtime": self.config.dna_hidden_twin_parquet.stat().st_mtime
            if self.config.dna_hidden_twin_parquet.exists()
            else None,
            "synapse_topk_path": str(synapse_path),
            "synapse_topk_mtime": synapse_path.stat().st_mtime if synapse_path.exists() else None,
        }
        return json.dumps(signatures, sort_keys=True, ensure_ascii=False)

    def _load_response_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.config.enable_response_cache or self.config.response_cache_path is None:
            return {}
        path = self.config.response_cache_path
        if not path.exists():
            return {}
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cache, dict):
                return cache
        except Exception:
            return {}
        return {}

    def _save_response_cache(self, cache_obj: Dict[str, Dict[str, Any]]) -> None:
        if not self.config.enable_response_cache or self.config.response_cache_path is None:
            return
        path = self.config.response_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _ticker_file_stems(ticker: str) -> List[str]:
        text = str(ticker).strip()
        if not text:
            return []
        digits = re.sub(r"\D", "", text)
        candidates = [text]
        if digits:
            try:
                as_int = str(int(digits))
                candidates.extend([as_int.zfill(4), as_int.zfill(5)])
            except ValueError:
                pass
        out: List[str] = []
        for value in candidates:
            if value and value not in out:
                out.append(value)
        return out

    def _load_ticker_qa(self, ticker: str) -> pd.DataFrame:
        if not self.config.qa_per_ticker_root:
            return pd.DataFrame()

        cache_key = _normalize_ticker(ticker)
        if cache_key in self._qa_df_cache:
            return self._qa_df_cache[cache_key]

        root = self.config.qa_per_ticker_root
        if not root.exists():
            self._qa_df_cache[cache_key] = pd.DataFrame()
            return self._qa_df_cache[cache_key]

        qa_path: Optional[Path] = None
        for stem in self._ticker_file_stems(ticker):
            candidate = root / f"{stem}_finetune_qa.csv"
            if candidate.exists():
                qa_path = candidate
                break

        if qa_path is None:
            self._qa_df_cache[cache_key] = pd.DataFrame()
            return self._qa_df_cache[cache_key]

        try:
            df = pd.read_csv(qa_path).copy()
        except Exception:
            df = pd.DataFrame()
        self._qa_df_cache[cache_key] = df
        return df

    @staticmethod
    def _compact_answer(answer: str, language: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(answer or "").strip())
        if not cleaned:
            return ""
        cleaned = re.sub(r"^根據基金文件[:：]\s*", "", cleaned)
        chunks = [c.strip() for c in re.split(r"(?<=[\.\!\?。！？])\s+", cleaned) if c.strip()]
        concise = " ".join(chunks[:2]).strip() if chunks else cleaned
        if len(concise) <= 420:
            return concise
        # Try to trim at a natural sentence boundary before hard cutoff.
        cut = max(concise.rfind(". ", 0, 420), concise.rfind("。", 0, 420), concise.rfind("! ", 0, 420), concise.rfind("? ", 0, 420))
        if cut > 120:
            return concise[: cut + 1].strip()
        return concise[:420].rsplit(" ", 1)[0].strip()

    @staticmethod
    def _to_investor_plain(answer: str, language: str, max_chars: int = 240) -> str:
        text = re.sub(r"\s+", " ", str(answer or "").strip())
        if not text:
            return ""

        # Remove common legal boilerplate that harms readability.
        boilerplate_patterns = [
            r"under the section entitled [\"“][^\"”]+[\"”]",
            r"shall be deemed to be deleted and replaced with the following[:：]?",
            r"forms an integral part of[^\.]*",
            r"with the prior approval of the Promoter[^\.]*",
        ]
        for p in boilerplate_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        if language != "zh":
            text = text.replace("Sub-Fund", "ETF")
            text = text.replace("The Sub -Fund", "The ETF")
            text = text.replace("TraHK", "the ETF")
            text = text.replace("Objective and Investment Strategy Objective", "Investment objective")
            text = re.sub(r"\s+,", ",", text)
            if "Quick facts" in text:
                text = text.split("Quick facts", 1)[0].strip()
        else:
            text = text.replace("TraHK", "本ETF")

        text = re.sub(r"\"\s*\d+\.\s*", " ", text)
        text = re.sub(r"\(\.\s*$", "", text)
        text = re.sub(r"\s*\(\s*$", "", text)
        text = re.sub(r"\s*-\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -;:,")
        if len(text) <= max_chars:
            return text

        cut = max(text.rfind(". ", 0, max_chars), text.rfind("。", 0, max_chars))
        if cut > 80:
            return text[: cut + 1].strip()
        return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."

    @staticmethod
    def _rewrite_fact_answer_concise(answer: str, language: str) -> str:
        text = (answer or "").strip()
        if not text:
            return text
        lower = text.lower()

        if language != "zh":
            if "closely correspond to the performance of the hang seng index" in lower:
                return "The fund aims to closely correspond to the performance of the Hang Seng Index."
            if "tracks hang seng index" in lower or "track hang seng index" in lower:
                return "It tracks the Hang Seng Index."
            if "investment objective" in lower and "hang seng index" in lower:
                return "Its investment objective is to closely track the Hang Seng Index."
        return text

    def _direct_fact_answer(self, ticker: str, user_query: str, language: str) -> Optional[str]:
        if not self.config.enable_direct_qa_lookup:
            return None
        # Keep recommendation-style questions for synthesis generation,
        # but allow factual risk questions to match per-ticker QA.
        if _is_recommendation_query(user_query):
            return None

        df = self._load_ticker_qa(ticker)
        if df.empty or not {"question", "answer"}.issubset(df.columns):
            return None

        query_norm = _normalize_for_match(user_query)
        query_tokens = _tokens(user_query)
        query_is_risk = _is_risk_query(user_query)
        query_is_fee = _is_fee_query(user_query)
        query_is_dividend = _is_dividend_query(user_query)
        query_is_tracking = _is_tracking_query(user_query)

        candidates = df.copy()
        language_pool = [language]
        if language == "zh":
            language_pool.extend(["zh-HK", "zh", "en"])
        else:
            language_pool.extend(["en", "zh-HK"])
        if "language" in candidates.columns:
            mask = candidates["language"].fillna("").astype(str).isin(language_pool)
            if mask.any():
                candidates = candidates[mask].copy()

        if candidates.empty:
            return None

        scored_rows: List[tuple[float, pd.Series]] = []
        for _, row in candidates.iterrows():
            question = str(row.get("question", ""))
            if not question:
                continue
            source_tag = str(row.get("source_tag", "")).lower()
            answer_text = str(row.get("answer", ""))
            question_low = question.lower()
            answer_low = answer_text.lower()

            seq = SequenceMatcher(None, query_norm, _normalize_for_match(question)).ratio()
            question_tokens = _tokens(question)
            overlap = (
                len(query_tokens & question_tokens) / max(len(query_tokens | question_tokens), 1)
                if query_tokens and question_tokens
                else 0.0
            )
            score = 0.7 * seq + 0.3 * overlap

            # Intent-specific guards to reduce incorrect fact retrieval.
            if query_is_risk:
                question_is_risk = _contains_any(question, ["risk", "risks", "volatility", "風險", "波動"])
                tag_is_risk = "risk" in source_tag
                if not question_is_risk and not tag_is_risk:
                    continue
                if question_is_risk:
                    score += 0.14
                if tag_is_risk:
                    score += 0.14

            if query_is_fee:
                fee_hit = _contains_any(question, ["fee", "charge", "expense", "費用", "收費", "管理費"]) or "fees_charges" in source_tag
                if not fee_hit:
                    continue
                if _contains_any(answer_text, ["lot size", "trading lot size", "每手"]):
                    # Strong penalty to avoid wrong mapping (fee question -> lot size answer).
                    score -= 0.45
                score += 0.16

            if query_is_dividend:
                dividend_hit = _contains_any(question, ["dividend", "distribution", "yield", "派息", "股息", "分派"]) or "dividend" in source_tag
                if not dividend_hit:
                    continue
                score += 0.14

            if query_is_tracking:
                if _contains_any(question, ["track", "index", "benchmark", "objective", "追蹤", "指數", "目標"]):
                    score += 0.12

            # Additional low-quality controls.
            if len(answer_low.strip()) < 25:
                score -= 0.08
            if "authorized under section 104" in answer_low and query_is_risk:
                score -= 0.2

            scored_rows.append((score, row))

        if not scored_rows:
            return None
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored_rows[0]
        if best_row is None or best_score < 0.38:
            return None

        if query_is_fee:
            percent_value = None
            # Prefer explicit ongoing-charge percentages from any ticker QA row.
            fee_rows = candidates.copy()
            if "source_tag" in fee_rows.columns:
                fee_rows["source_tag"] = fee_rows["source_tag"].fillna("").astype(str).str.lower()
            else:
                fee_rows["source_tag"] = ""
            fee_rows["answer_text"] = fee_rows["answer"].fillna("").astype(str)
            fee_rows["fee_hint"] = fee_rows.apply(
                lambda r: (
                    ("fees_charges" in r["source_tag"])
                    or _contains_any(r["answer_text"], ["ongoing charges", "expense ratio", "management fee", "費用", "收費", "管理費"])
                ),
                axis=1,
            )
            fee_candidates = fee_rows[fee_rows["fee_hint"]].copy()
            if fee_candidates.empty:
                fee_candidates = fee_rows

            for _, row in fee_candidates.iterrows():
                ans = str(row.get("answer_text", ""))
                for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", ans):
                    value = float(m.group(1))
                    # Keep realistic fee-like percentages.
                    if 0.0 < value <= 5.0:
                        percent_value = f"{value:.3f}".rstrip("0").rstrip(".")
                        break
                if percent_value is not None:
                    break

            if language == "zh":
                if percent_value is not None:
                    return f"根據基金文件，這隻ETF的全年持續收費約為 {percent_value}%。另外，交易及申購/贖回可能涉及其他費用。"
                return "根據基金文件，這隻ETF會有持續收費及交易相關費用；具體收費請以最新產品資料概要及基金章程為準。"
            if percent_value is not None:
                return (
                    f"Based on fund documents, ongoing charges are about {percent_value}% per year. "
                    "Additional dealing/creation-redemption fees may also apply."
                )
            return (
                "Based on fund documents, this ETF has ongoing charges plus transaction "
                "and creation/redemption related fees; please refer to the latest KFS/prospectus for exact breakdown."
            )

        # For risk questions, return a more comprehensive multi-point answer.
        if query_is_risk:
            top_risk_answers: List[str] = []
            seen_norm: set[str] = set()
            for score, row in scored_rows[:8]:
                if score < 0.30:
                    continue
                source_tag = str(row.get("source_tag", "")).lower()
                question = str(row.get("question", ""))
                if "risk" not in source_tag and not _contains_any(question, ["risk", "risks", "volatility", "風險", "波動"]):
                    continue
                ans = self._compact_answer(str(row.get("answer", "")), language=language)
                # Skip noisy legal addendum boilerplate fragments when better options exist.
                if _contains_any(ans, ["sub-section entitled", "forms an integral part", "addendum dated"]):
                    continue
                ans = self._to_investor_plain(ans, language=language, max_chars=220)
                norm = _normalize_for_match(ans)
                if not ans or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                top_risk_answers.append(ans)
                if len(top_risk_answers) >= 3:
                    break

            if top_risk_answers:
                if language == "zh":
                    bullets = "\n".join([f"- {a}" for a in top_risk_answers])
                    return (
                        "根據基金文件，主要風險包括：\n"
                        f"{bullets}\n"
                        "這代表甚麼：若你已集中港股或單一主題，宜搭配不同地區或資產類別ETF分散波動。"
                    )
                bullets = "\n".join([f"- {a}" for a in top_risk_answers])
                return (
                    "Based on fund documents, the key risks include:\n"
                    f"{bullets}\n"
                    "What this means: if your portfolio is concentrated in one market/theme, consider adding other regions or asset classes to diversify volatility."
                )

        answer = self._compact_answer(str(best_row.get("answer", "")), language=language)
        if not answer:
            return None
        answer = self._rewrite_fact_answer_concise(answer, language=language)
        if language == "zh":
            return f"根據基金文件，{answer}"
        return answer

    def _classify_user_intent(self, user_query: str) -> str:
        text = user_query or ""
        if _is_model1_related_query(text):
            return "model1_related"
        if _is_news_discovery_query(text):
            return "model2_discovery"
        if _is_feature_query(text):
            return "etf_features"
        if _is_fee_query(text) or _is_dividend_query(text) or _is_tracking_query(text):
            return "factual"
        if _is_risk_query(text):
            return "risk_explain"
        return "general"

    def _answer_model1_related(self, ticker: str, dna_context: Dict[str, Any], language: str) -> str:
        alts = dna_context.get("top_3_alternatives", []) or []
        cluster_id = dna_context.get("cluster_id")
        notes = str(dna_context.get("notes", "") or "")
        if not alts:
            if language == "zh":
                if notes:
                    return (
                        f"目前未能提供 `{_normalize_ticker(ticker)}` 的Model 1相關ETF候選。\n"
                        f"原因：{notes}\n"
                        "請先確認DNA輸出（cluster/advisory parquet）是否可讀，之後即可返回同群組替代ETF。"
                    )
                return f"未找到 `{_normalize_ticker(ticker)}` 的相關ETF候選。請先確認DNA輸出是否完整。"
            if notes:
                return (
                    f"Model 1 related-ticker lookup is currently unavailable for `{_normalize_ticker(ticker)}`.\n"
                    f"Reason: {notes}\n"
                    "Please verify DNA cluster/advisory parquet outputs, then retry."
                )
            return f"No related ETF candidates were found for `{_normalize_ticker(ticker)}`. Please verify DNA outputs."

        lines = []
        for i, alt in enumerate(alts, start=1):
            t = str(alt.get("ticker", ""))
            signal = str(alt.get("signal", "similar_profile"))
            dist = alt.get("pc_distance")
            dist_text = f"{float(dist):.4f}" if isinstance(dist, (int, float)) else "N/A"
            lines.append((i, t, signal, dist_text))

        if language == "zh":
            body = "\n".join([f"- {i}. `{t}`（signal: {s}, distance: {d}）" for i, t, s, d in lines])
            return (
                f"根據Model 1（Financial DNA）聚類結果，`{_normalize_ticker(ticker)}` "
                f"目前的cluster_id為 `{cluster_id}`。可優先參考以下相近ETF：\n{body}\n"
                "建議：先比較費用率、資產類別及地域曝險，再決定是否替換或分散配置。"
            )
        body = "\n".join([f"- {i}. `{t}` (signal: {s}, distance: {d})" for i, t, s, d in lines])
        return (
            f"Based on Model 1 (Financial DNA), `{_normalize_ticker(ticker)}` is in cluster `{cluster_id}`. "
            f"Closest related ETF candidates are:\n{body}\n"
            "Practical advice: compare fee level, asset-class exposure, and geographic concentration before switching."
        )

    def _discover_related_tickers_from_synapse(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        syn = self._load_synapse_topk().copy()
        if syn.empty or "predicted_ticker" not in syn.columns:
            return []

        text_cols = [c for c in ["Headline", "Market_Event", "query_text", "Sector", "Source"] if c in syn.columns]
        if not text_cols:
            return []

        q_tokens = _tokens(user_query)
        if not q_tokens:
            return []

        syn["final_score"] = pd.to_numeric(syn.get("final_score"), errors="coerce").fillna(0.0)

        def _row_score(row: pd.Series) -> float:
            content = " ".join(str(row.get(c, "")) for c in text_cols)
            r_tokens = _tokens(content)
            if not r_tokens:
                return 0.0
            overlap = len(q_tokens & r_tokens) / max(len(q_tokens), 1)
            return 0.65 * overlap + 0.35 * float(row.get("final_score", 0.0))

        syn["query_match_score"] = syn.apply(_row_score, axis=1)
        syn = syn[syn["predicted_ticker"].astype(str).str.len() > 0].copy()
        if syn.empty:
            return []

        top = (
            syn.sort_values(["query_match_score", "final_score"], ascending=[False, False])
            .groupby("predicted_ticker", as_index=False)
            .head(1)
            .sort_values(["query_match_score", "final_score"], ascending=[False, False])
            .head(top_k)
        )
        out: List[Dict[str, Any]] = []
        for _, row in top.iterrows():
            out.append(
                {
                    "ticker": _normalize_ticker(row.get("predicted_ticker")),
                    "headline": str(row.get("Headline", ""))[:160],
                    "market_event": str(row.get("Market_Event", ""))[:120],
                    "score": float(row.get("query_match_score", 0.0)),
                }
            )
        return out

    def _answer_model2_discovery(self, user_query: str, language: str) -> str:
        matches = self._discover_related_tickers_from_synapse(user_query=user_query, top_k=3)
        if not matches:
            if language == "zh":
                return "未能從Model 2（Synapse）找到明確相關ETF。請提供更多新聞關鍵字、事件描述或市場主題。"
            return "I could not find confident related ETFs from Model 2 (Synapse). Please provide more specific news/event keywords."

        if language == "zh":
            lines = []
            for i, m in enumerate(matches, start=1):
                lines.append(
                    f"- {i}. `{m['ticker']}`：事件 `{m['market_event'] or 'N/A'}`；摘要 `{m['headline'] or 'N/A'}`"
                )
            return (
                "根據Model 2（Synapse）語義匹配，以下ETF與你提供的資訊最相關：\n"
                + "\n".join(lines)
                + "\n建議：先點選其中1-2隻查看其持倉與1年走勢，再進一步比較風險與費用。"
            )

        lines = []
        for i, m in enumerate(matches, start=1):
            lines.append(
                f"- {i}. `{m['ticker']}`: event `{m['market_event'] or 'N/A'}`; headline summary `{m['headline'] or 'N/A'}`"
            )
        return (
            "Based on Model 2 (Synapse) semantic matching, these ETFs are most related to your information:\n"
            + "\n".join(lines)
            + "\nPractical next step: open 1-2 of these tickers, then compare holdings, 1Y trend, and fee profile."
        )

    def _answer_etf_features_with_advice(self, ticker: str, user_query: str, language: str) -> Optional[str]:
        df = self._load_ticker_qa(ticker)
        if df.empty or not {"source_tag", "answer"}.issubset(df.columns):
            return None

        if "language" in df.columns:
            if language == "zh":
                lang_candidates = ["zh-HK", "zh", "en"]
            else:
                lang_candidates = ["en", "zh-HK"]
            lang_mask = df["language"].fillna("").astype(str).isin(lang_candidates)
            if lang_mask.any():
                df = df[lang_mask].copy()

        want_tags = ["objective_strategy", "fees_charges", "dividend", "key_risks", "currency_counter"]
        selected: Dict[str, str] = {}
        for tag in want_tags:
            sub = df[df["source_tag"].fillna("").astype(str).str.contains(tag, case=False, regex=False)]
            if sub.empty:
                continue
            ans = self._compact_answer(str(sub.iloc[0]["answer"]), language=language)
            ans = self._to_investor_plain(ans, language=language, max_chars=210)
            if ans:
                selected[tag] = ans

        if not selected:
            return None

        if language == "zh":
            lines = []
            if "objective_strategy" in selected:
                lines.append(f"- 投資目標：{selected['objective_strategy']}")
            if "fees_charges" in selected:
                lines.append(f"- 費用重點：{selected['fees_charges']}")
            if "dividend" in selected:
                lines.append(f"- 派息資訊：{selected['dividend']}")
            if "key_risks" in selected:
                lines.append(f"- 風險重點：{selected['key_risks']}")
            if "currency_counter" in selected:
                lines.append(f"- 交易櫃台/貨幣：{selected['currency_counter']}")
            return (
                f"`{_normalize_ticker(ticker)}` 的ETF特點如下（根據文件）：\n"
                + "\n".join(lines)
                + "\n\n建議組合做法：可先把此ETF作為核心或衛星倉位，再搭配不同地區/資產類別ETF做分散，避免單一市場集中。"
            )

        lines = []
        if "objective_strategy" in selected:
            lines.append(f"- Objective: {selected['objective_strategy']}")
        if "fees_charges" in selected:
            lines.append(f"- Fee profile: {selected['fees_charges']}")
        if "dividend" in selected:
            lines.append(f"- Dividend: {selected['dividend']}")
        if "key_risks" in selected:
            lines.append(f"- Risk highlights: {selected['key_risks']}")
        if "currency_counter" in selected:
            lines.append(f"- Trading/currency counters: {selected['currency_counter']}")
        return (
            f"Key ETF features for `{_normalize_ticker(ticker)}` (from fund documents):\n"
            + "\n".join(lines)
            + "\n\nPortfolio construction advice: treat this ETF as a core or satellite position, then add ETFs from different regions/asset classes to reduce single-market concentration."
        )

    def _get_sentence_encoder(self):
        if self._sentence_encoder is not None:
            return self._sentence_encoder
        from sentence_transformers import SentenceTransformer  # lazy import

        self._sentence_encoder = SentenceTransformer(self.config.sentence_model_name)
        return self._sentence_encoder

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # Requirement 1
    def get_synthesis_context(self, ticker: str) -> Dict[str, Any]:
        ticker_norm = _normalize_ticker(ticker)
        cluster_df = self._load_dna_cluster()
        home_bias_df = self._load_home_bias()
        hidden_twin_df = self._load_hidden_twin()

        ticker_rows = cluster_df[cluster_df["ticker"] == ticker_norm].copy()
        if ticker_rows.empty:
            return {
                "ticker": ticker_norm,
                "cluster_id": None,
                "cluster_by_perspective": {},
                "top_3_alternatives": [],
                "notes": (
                    "Ticker missing from current DNA cluster artifacts "
                    "(coverage gap). Do not treat this as invalid/non-existent ticker."
                ),
            }

        cluster_by_perspective = {}
        if {"perspective", "cluster_id"}.issubset(ticker_rows.columns):
            sub = ticker_rows[["perspective", "cluster_id"]].dropna().drop_duplicates()
            cluster_by_perspective = {str(row["perspective"]): int(row["cluster_id"]) for _, row in sub.iterrows()}

        dominant_cluster = None
        if "cluster_id" in ticker_rows.columns:
            modes = ticker_rows["cluster_id"].dropna().mode()
            if not modes.empty:
                dominant_cluster = int(modes.iloc[0])

        candidates: List[Dict[str, Any]] = []

        if not home_bias_df.empty:
            hb = home_bias_df[home_bias_df.get("source_ticker", "") == ticker_norm]
            for _, row in hb.iterrows():
                candidates.append(
                    {
                        "ticker": _normalize_ticker(row.get("alternative_ticker", "")),
                        "signal": str(row.get("signal", "home_bias_candidate")),
                        "pc_distance": float(row.get("pc_distance", 999.0)),
                        "perspective": row.get("perspective"),
                    }
                )

        if not hidden_twin_df.empty:
            ht_a = hidden_twin_df[hidden_twin_df.get("ticker_a", "") == ticker_norm].copy()
            ht_a["alt"] = ht_a["ticker_b"]
            ht_b = hidden_twin_df[hidden_twin_df.get("ticker_b", "") == ticker_norm].copy()
            ht_b["alt"] = ht_b["ticker_a"]
            ht = pd.concat([ht_a, ht_b], ignore_index=True) if (not ht_a.empty or not ht_b.empty) else pd.DataFrame()
            for _, row in ht.iterrows():
                candidates.append(
                    {
                        "ticker": _normalize_ticker(row.get("alt", "")),
                        "signal": str(row.get("signal", "hidden_twin_candidate")),
                        "pc_distance": float(row.get("pc_distance", 999.0)),
                        "perspective": row.get("perspective"),
                    }
                )

        alt_df = pd.DataFrame(candidates)
        if not alt_df.empty:
            alt_df = alt_df[alt_df["ticker"] != ""]
            alt_df = (
                alt_df.sort_values(["pc_distance"])
                .drop_duplicates(subset=["ticker"], keep="first")
                .head(self.config.top_alt_k)
            )
            top_3_alternatives = alt_df.to_dict(orient="records")
        else:
            top_3_alternatives = []

        return {
            "ticker": ticker_norm,
            "cluster_id": dominant_cluster,
            "cluster_by_perspective": cluster_by_perspective,
            "top_3_alternatives": top_3_alternatives,
        }

    # Requirement 2
    def get_synapse_alerts(self, ticker: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
        ticker_norm = _normalize_ticker(ticker)
        syn = self._load_synapse_topk().copy()

        required_cols = {"predicted_ticker", "final_score"}
        missing = [c for c in required_cols if c not in syn.columns]
        if missing:
            raise ValueError(f"Synapse top-k output missing required columns: {missing}")

        subset = syn[
            (syn["predicted_ticker"] == ticker_norm)
            & (pd.to_numeric(syn["final_score"], errors="coerce") >= self.config.similarity_threshold)
        ].copy()
        if subset.empty:
            return []

        subset["final_score"] = pd.to_numeric(subset["final_score"], errors="coerce")
        if "Date" in subset.columns:
            subset = subset.sort_values(["Date", "final_score"], ascending=[False, False])
        else:
            subset = subset.sort_values(["final_score"], ascending=[False])

        keep_cols = [
            c
            for c in [
                "Date",
                "Headline",
                "Source",
                "Market_Event",
                "Sector",
                "final_score",
                "query_text",
            ]
            if c in subset.columns
        ]
        out = (
            subset[keep_cols]
            .drop_duplicates(subset=[c for c in ["Date", "Headline"] if c in keep_cols])
            .head(max(self.config.max_alerts * 2, 16))
            .copy()
        )
        out = out.rename(columns={"final_score": "similarity_score"})

        if query and self.config.enable_query_similarity and "Headline" in out.columns:
            try:
                encoder = self._get_sentence_encoder()
                query_emb = encoder.encode([query], normalize_embeddings=True)[0]
                heads = out["Headline"].fillna("").astype(str).tolist()
                head_emb = encoder.encode(heads, normalize_embeddings=True)
                out["query_similarity"] = [self._cosine(query_emb, emb) for emb in head_emb]
                out = out.sort_values(["query_similarity", "similarity_score"], ascending=[False, False])
            except Exception:
                # If real-time embedding fails, continue with Synapse ranking only.
                pass

        out = out.head(self.config.max_alerts)
        records = out.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("Date"), pd.Timestamp):
                r["Date"] = r["Date"].date().isoformat()
        return records

    def _build_system_message(self, language: str) -> str:
        if language == "zh":
            return (
                "你是香港ETF資訊助理（教育用途，不構成投資建議）。\n"
                "核心原則：先回答用戶問題本身，準確、簡潔、友善。\n"
                "不要主動推薦新ETF，也不要主動講風險；只有用戶明確問到時才提供。\n"
                "若是事實型問題（例如ETF追蹤什麼、成分、類型），只提供事實答案與必要背景。\n"
                "目標ETF代號由系統提供並視為有效上下文；不要自行斷言該代號『不存在』或『無效』。\n"
                "若資料不足，請明確說明『目前本地資料不足以確認細節』，並建議查閱該ETF官方KFS/章程。\n"
                "使用香港常用金融術語與清晰段落，避免離題。"
            )
        return (
            "You are an HK ETF information assistant (educational only, not financial advice).\n"
            "Primary rule: answer the user's actual question first, clearly and politely.\n"
            "Do NOT proactively recommend other ETFs or discuss risk unless the user explicitly asks.\n"
            "For factual questions (e.g., what an ETF tracks), provide direct facts and brief context only.\n"
            "The target ETF ticker is provided by the system and should be treated as valid context; do not claim it is invalid or non-existent.\n"
            "If details are missing in local data, say data is currently insufficient and suggest checking official ETF KFS/prospectus.\n"
            "Use clear HK-market terminology and avoid going beyond the user intent."
        )

    def _build_user_prompt(
        self,
        user_query: str,
        ticker: str,
        dna_context: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        language: str,
    ) -> str:
        header = "使用者問題" if language == "zh" else "User Query"
        ticker_label = "目標ETF" if language == "zh" else "Target ETF"
        ctx_label = "Financial DNA Context"
        syn_label = "Synapse Alerts"
        risk_asked = _is_risk_query(user_query)
        recommendation_asked = _is_recommendation_query(user_query)
        intent_rules = {
            "answer_mode": "information_first",
            "risk_discussion_allowed": risk_asked,
            "recommendation_allowed": recommendation_asked,
            "policy_version": PROMPT_POLICY_VERSION,
        }
        prompt_dna_context = dict(dna_context or {})
        notes_text = str(prompt_dna_context.get("notes", "") or "").lower()
        if "coverage gap" in notes_text or "ticker not found" in notes_text:
            # Do not leak artifact-coverage wording that can mislead the model into
            # claiming the ticker is invalid; keep context neutral.
            prompt_dna_context["notes"] = "DNA cluster artifacts unavailable for this ticker; continue with ticker facts."
        return (
            f"{header}: {user_query}\n"
            f"{ticker_label}: {ticker}\n\n"
            "Important Guardrail:\n"
            "- The target ticker is system-provided context and should be treated as valid.\n"
            "- If DNA/Synapse context is missing, describe it as local-data coverage gap only.\n"
            "- Do NOT claim the ticker is invalid, non-existent, or absent from the market/database.\n\n"
            f"Intent Rules:\n{json.dumps(intent_rules, ensure_ascii=False, indent=2)}\n\n"
            f"{ctx_label}:\n{json.dumps(prompt_dna_context, ensure_ascii=False, indent=2)}\n\n"
            f"{syn_label}:\n{json.dumps(alerts, ensure_ascii=False, indent=2)}\n\n"
            "Please answer in an information-first, user-friendly style."
        )

    @staticmethod
    def _is_low_quality_response(text: str) -> bool:
        content = (text or "").strip()
        if len(content) < 24:
            return True
        if content in {"#", "##", "###", "-", "--", "..."}:
            return True
        # Reject outputs that are mostly markup characters without advisory text.
        non_alnum = sum(1 for ch in content if not ch.isalnum() and ch not in {" ", "\n"})
        return non_alnum / max(len(content), 1) > 0.6

    def _build_fallback_response(
        self,
        ticker: str,
        language: str,
        dna_context: Dict[str, Any],
        user_query: str,
    ) -> str:
        risk_asked = _is_risk_query(user_query)
        recommendation_asked = _is_recommendation_query(user_query)
        alt_tickers = [x.get("ticker") for x in dna_context.get("top_3_alternatives", []) if x.get("ticker")]
        if language == "zh":
            base = (
                f"你查詢的是 {_normalize_ticker(ticker)}。目前本地模型未能即時生成完整回答，"
                "先提供精簡事實答覆：此ETF的具體追蹤指數/標的請以基金名稱與產品概要（KFS/Prospectus）為準。"
            )
            if recommendation_asked and alt_tickers:
                return base + f"\n你有問到替代選項，可先參考：{'、'.join(alt_tickers[:3])}。"
            if risk_asked:
                return base + "\n你有問到風險，重點可先看：市場波動、行業集中度與追蹤誤差。"
            return base

        base = (
            f"You asked about {_normalize_ticker(ticker)}. The local model could not produce a full response in time, "
            "so here is a concise factual fallback: confirm the ETF's exact benchmark/index in the fund name and official KFS/prospectus."
        )
        if recommendation_asked and alt_tickers:
            return base + f" Since you asked for recommendations, you can review: {', '.join(alt_tickers[:3])}."
        if risk_asked:
            return base + " Since you asked about risk, check market volatility, concentration, and tracking-error exposure."
        return base

    def _run_qwen_ollama(
        self,
        user_prompt: str,
        system_message: str,
        model_name: str,
        endpoint: str = "http://localhost:11434/api/generate",
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model_name,
            "system": system_message,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(endpoint, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()

    def _run_qwen_vllm(
        self,
        user_prompt: str,
        system_message: str,
        model_name: str,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        resp = requests.post(endpoint, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _run_qwen_transformers(
        self,
        user_prompt: str,
        system_message: str,
        model_name: str,
        max_new_tokens: int = 220,
    ) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._ensure_transformers_loaded(model_name=model_name)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        model_device = next(self._hf_model.parameters()).device
        if hasattr(self._hf_tokenizer, "apply_chat_template"):
            inputs = self._hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model_device)
        else:
            prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(model_device)

        eos_token_id = self._hf_tokenizer.eos_token_id
        end_of_turn_id = None
        if hasattr(self._hf_tokenizer, "convert_tokens_to_ids"):
            try:
                end_of_turn_id = self._hf_tokenizer.convert_tokens_to_ids("<|im_end|>")
            except Exception:
                end_of_turn_id = None
        eos_ids = [tid for tid in [eos_token_id, end_of_turn_id] if isinstance(tid, int) and tid >= 0]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "max_time": 45.0,
        }
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
            gen_kwargs["pad_token_id"] = eos_ids[0]
        elif eos_token_id is not None:
            gen_kwargs["pad_token_id"] = eos_token_id

        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                outputs = self._hf_model.generate(inputs, **gen_kwargs)
                generated = outputs[0][inputs.shape[-1] :]
                text = self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
            else:
                outputs = self._hf_model.generate(**inputs, **gen_kwargs)
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[0][input_len:]
                text = self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text

    def _ensure_transformers_loaded(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._hf_model is not None and self._hf_tokenizer is not None:
            return

        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_cuda = torch.cuda.is_available()
        use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        dtype = torch.float16 if (use_cuda or use_mps) else torch.float32
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            model_kwargs["device_map"] = "auto"
        self._hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not use_cuda and use_mps:
            self._hf_model = self._hf_model.to("mps")
        elif not use_cuda:
            self._hf_model = self._hf_model.to("cpu")

    def warmup_model(self, backend: str = "transformers", qwen_model: str = DEFAULT_LOCAL_QWEN_MODEL) -> Dict[str, Any]:
        """Load model/resources eagerly so first user request is responsive."""
        if backend == "transformers":
            self._ensure_transformers_loaded(model_name=qwen_model)
            return {"ok": True, "backend": backend, "model": qwen_model, "warmed": True}
        if backend in {"ollama", "vllm"}:
            # HTTP backends run outside this process; no local model warmup needed.
            return {"ok": True, "backend": backend, "model": qwen_model, "warmed": False}
        raise ValueError(f"Unsupported backend: {backend}. Use one of: ollama, vllm, transformers")

    def _run_with_timeout(self, fn, timeout_seconds: int, *args, **kwargs):
        """
        Execute a callable with hard wall-clock timeout on Unix.
        Raises TimeoutError when limit is hit.
        """
        if timeout_seconds <= 0:
            return fn(*args, **kwargs)

        # Streamlit script execution may happen in a worker thread where
        # signal handlers are disallowed. In that case, execute directly.
        if threading.current_thread() is not threading.main_thread():
            return fn(*args, **kwargs)

        if not hasattr(signal, "SIGALRM"):
            # Fallback for platforms without SIGALRM support.
            return fn(*args, **kwargs)

        def _handler(signum, frame):
            raise TimeoutError(f"LLM inference exceeded {timeout_seconds}s timeout.")

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def synthesize_response(
        self,
        ticker: str,
        user_query: str,
        backend: str = "transformers",
        qwen_model: str = DEFAULT_LOCAL_QWEN_MODEL,
    ) -> Dict[str, Any]:
        cache_key = self._response_cache_key(
            ticker=ticker,
            user_query=user_query,
            backend=backend,
            qwen_model=qwen_model,
        )
        cache_obj = self._load_response_cache()
        cached = cache_obj.get(cache_key)
        if cached is not None:
            cached_text = str(cached.get("response", "")).strip()
            if self._is_low_quality_response(cached_text):
                cache_obj.pop(cache_key, None)
                self._save_response_cache(cache_obj)
            else:
                return {
                    **cached,
                    "cache_hit": True,
                }

        language = _detect_language(user_query)
        intent = self._classify_user_intent(user_query)

        if intent == "model2_discovery":
            model2_answer = self._answer_model2_discovery(user_query=user_query, language=language)
            result = {
                "response": model2_answer,
                "data_evidence": {
                    "ticker": _normalize_ticker(ticker),
                    "cluster_id": None,
                    "top_3_alternatives": [],
                    "news_similarity_scores": [],
                    "answer_mode": "model2_ticker_discovery",
                    "intent": intent,
                },
                "language": "zh-HK" if language == "zh" else "en",
                "backend": backend,
                "model": qwen_model,
                "cache_hit": False,
            }
            cache_obj[cache_key] = {**result, "cached_at": datetime.now().isoformat()}
            self._save_response_cache(cache_obj)
            return result

        # Fast path: allow direct QA answer even if DNA/Synapse artifacts are unavailable.
        direct_answer = None
        if intent in {"factual", "risk_explain", "general"}:
            direct_answer = self._direct_fact_answer(ticker=ticker, user_query=user_query, language=language)
        if direct_answer:
            evidence_block = {
                "ticker": _normalize_ticker(ticker),
                "cluster_id": None,
                "top_3_alternatives": [],
                "news_similarity_scores": [],
                "answer_mode": "direct_qa_lookup",
                "intent": intent,
            }
            result = {
                "response": direct_answer,
                "data_evidence": evidence_block,
                "language": "zh-HK" if language == "zh" else "en",
                "backend": backend,
                "model": qwen_model,
                "cache_hit": False,
            }
            cache_obj[cache_key] = {**result, "cached_at": datetime.now().isoformat()}
            self._save_response_cache(cache_obj)
            return result

        try:
            dna_context = self.get_synthesis_context(ticker)
        except Exception as exc:
            dna_context = {
                "ticker": _normalize_ticker(ticker),
                "cluster_id": None,
                "cluster_by_perspective": {},
                "top_3_alternatives": [],
                "notes": f"DNA context unavailable: {exc}",
            }

        if intent == "model1_related":
            model1_answer = self._answer_model1_related(ticker=ticker, dna_context=dna_context, language=language)
            result = {
                "response": model1_answer,
                "data_evidence": {
                    "ticker": _normalize_ticker(ticker),
                    "cluster_id": dna_context.get("cluster_id"),
                    "top_3_alternatives": dna_context.get("top_3_alternatives", []),
                    "news_similarity_scores": [],
                    "answer_mode": "model1_related_tickers",
                    "intent": intent,
                },
                "language": "zh-HK" if language == "zh" else "en",
                "backend": backend,
                "model": qwen_model,
                "cache_hit": False,
            }
            cache_obj[cache_key] = {**result, "cached_at": datetime.now().isoformat()}
            self._save_response_cache(cache_obj)
            return result

        if intent == "etf_features":
            feature_answer = self._answer_etf_features_with_advice(ticker=ticker, user_query=user_query, language=language)
            if feature_answer:
                result = {
                    "response": feature_answer,
                    "data_evidence": {
                        "ticker": _normalize_ticker(ticker),
                        "cluster_id": dna_context.get("cluster_id"),
                        "top_3_alternatives": dna_context.get("top_3_alternatives", []),
                        "news_similarity_scores": [],
                        "answer_mode": "etf_features_with_advice",
                        "intent": intent,
                    },
                    "language": "zh-HK" if language == "zh" else "en",
                    "backend": backend,
                    "model": qwen_model,
                    "cache_hit": False,
                }
                cache_obj[cache_key] = {**result, "cached_at": datetime.now().isoformat()}
                self._save_response_cache(cache_obj)
                return result

        try:
            synapse_alerts = self.get_synapse_alerts(ticker=ticker, query=user_query)
        except Exception as exc:
            synapse_alerts = [{"Date": None, "Headline": "Synapse alerts unavailable", "similarity_score": None, "query_similarity": None, "error": str(exc)}]

        system_message = self._build_system_message(language)
        user_prompt = self._build_user_prompt(
            user_query=user_query,
            ticker=ticker,
            dna_context=dna_context,
            alerts=synapse_alerts,
            language=language,
        )

        if backend == "ollama":
            try:
                ai_response = self._run_qwen_ollama(user_prompt, system_message, model_name=qwen_model)
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(
                    "Ollama endpoint is not reachable on localhost:11434. "
                    "Use fully local pretrained inference via backend='transformers' "
                    f"with model='{DEFAULT_LOCAL_QWEN_MODEL}', or start Ollama first."
                ) from exc
        elif backend == "vllm":
            try:
                ai_response = self._run_qwen_vllm(user_prompt, system_message, model_name=qwen_model)
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(
                    "vLLM endpoint is not reachable on localhost:8000. "
                    "Use backend='transformers' for in-process local inference."
                ) from exc
        elif backend == "transformers":
            try:
                ai_response = self._run_with_timeout(
                    self._run_qwen_transformers,
                    self.config.llm_timeout_seconds,
                    user_prompt,
                    system_message,
                    qwen_model,
                )
            except Exception:
                # Always return a practical fallback instead of hanging or failing hard.
                ai_response = self._build_fallback_response(
                    ticker=ticker,
                    language=language,
                    dna_context=dna_context,
                    user_query=user_query,
                )
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use one of: ollama, vllm, transformers")

        if self._is_low_quality_response(ai_response):
            ai_response = self._build_fallback_response(
                ticker=ticker,
                language=language,
                dna_context=dna_context,
                user_query=user_query,
            )

        evidence_block = {
            "ticker": _normalize_ticker(ticker),
            "cluster_id": dna_context.get("cluster_id"),
            "top_3_alternatives": dna_context.get("top_3_alternatives", []),
            "intent": intent,
            "news_similarity_scores": [
                {
                    "date": item.get("Date"),
                    "headline": item.get("Headline"),
                    "synapse_similarity_score": item.get("similarity_score"),
                    "query_similarity": item.get("query_similarity"),
                }
                for item in synapse_alerts
            ],
        }

        result = {
            "response": ai_response,
            "data_evidence": evidence_block,
            "language": "zh-HK" if language == "zh" else "en",
            "backend": backend,
            "model": qwen_model,
            "cache_hit": False,
        }
        cache_obj[cache_key] = {**result, "cached_at": datetime.now().isoformat()}
        self._save_response_cache(cache_obj)
        return result


_DEFAULT_ENGINE = SynthesisEngine()
_ENGINE_VARIANTS: Dict[tuple[Optional[bool], Optional[bool]], SynthesisEngine] = {}


def get_synthesis_context(ticker: str) -> Dict[str, Any]:
    """Required public function: pull cluster ID + top-3 alternatives from DNA outputs."""
    return _DEFAULT_ENGINE.get_synthesis_context(ticker)


def get_synapse_alerts(ticker: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Required public function: pull recent Synapse alerts above similarity threshold."""
    return _DEFAULT_ENGINE.get_synapse_alerts(ticker=ticker, query=query)


def generate_synthesis(
    ticker: str,
    user_query: str,
    backend: str = "transformers",
    qwen_model: str = DEFAULT_LOCAL_QWEN_MODEL,
    enable_query_similarity: Optional[bool] = None,
    enable_response_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    """Main interface for app integration."""
    engine = _DEFAULT_ENGINE
    if enable_query_similarity is not None or enable_response_cache is not None:
        variant_key = (enable_query_similarity, enable_response_cache)
        if variant_key in _ENGINE_VARIANTS:
            engine = _ENGINE_VARIANTS[variant_key]
        else:
            cfg = SynthesisConfig.default()
            if enable_query_similarity is not None:
                cfg.enable_query_similarity = enable_query_similarity
            if enable_response_cache is not None:
                cfg.enable_response_cache = enable_response_cache
            engine = SynthesisEngine(cfg)
            _ENGINE_VARIANTS[variant_key] = engine

    return engine.synthesize_response(
        ticker=ticker,
        user_query=user_query,
        backend=backend,
        qwen_model=qwen_model,
    )


def warmup_synthesis_model(
    backend: str = "transformers",
    qwen_model: str = DEFAULT_LOCAL_QWEN_MODEL,
    enable_query_similarity: Optional[bool] = None,
    enable_response_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    """Public app helper: initialize synthesis model/resources in advance."""
    engine = _DEFAULT_ENGINE
    if enable_query_similarity is not None or enable_response_cache is not None:
        variant_key = (enable_query_similarity, enable_response_cache)
        if variant_key in _ENGINE_VARIANTS:
            engine = _ENGINE_VARIANTS[variant_key]
        else:
            cfg = SynthesisConfig.default()
            if enable_query_similarity is not None:
                cfg.enable_query_similarity = enable_query_similarity
            if enable_response_cache is not None:
                cfg.enable_response_cache = enable_response_cache
            engine = SynthesisEngine(cfg)
            _ENGINE_VARIANTS[variant_key] = engine
    return engine.warmup_model(backend=backend, qwen_model=qwen_model)
