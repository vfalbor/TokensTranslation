"""
Token counting using tiktoken (cl100k_base — same as GPT-4 and close to Claude's BPE).
Provides per-language token efficiency analysis.
"""
import tiktoken
from functools import lru_cache
from typing import Dict

_ENCODING = None

LANGUAGE_TOKEN_MULTIPLIERS: Dict[str, Dict] = {
    "en": {"name": "English",      "multiplier": 1.00, "flag": "🇬🇧", "efficiency": "Baseline"},
    "zh": {"name": "Chinese",      "multiplier": 0.92, "flag": "🇨🇳", "efficiency": "Excellent — dense meaning per token"},
    "ja": {"name": "Japanese",     "multiplier": 1.05, "flag": "🇯🇵", "efficiency": "Good — kanji is dense"},
    "de": {"name": "German",       "multiplier": 1.35, "flag": "🇩🇪", "efficiency": "Moderate — compound words split"},
    "fr": {"name": "French",       "multiplier": 1.28, "flag": "🇫🇷", "efficiency": "Moderate — accented chars"},
    "es": {"name": "Spanish",      "multiplier": 1.27, "flag": "🇪🇸", "efficiency": "Moderate — morphology overhead"},
    "pt": {"name": "Portuguese",   "multiplier": 1.30, "flag": "🇵🇹", "efficiency": "Moderate"},
    "it": {"name": "Italian",      "multiplier": 1.25, "flag": "🇮🇹", "efficiency": "Moderate"},
    "nl": {"name": "Dutch",        "multiplier": 1.22, "flag": "🇳🇱", "efficiency": "Moderate"},
    "ru": {"name": "Russian",      "multiplier": 1.85, "flag": "🇷🇺", "efficiency": "Poor — Cyrillic 2-byte overhead"},
    "ar": {"name": "Arabic",       "multiplier": 2.10, "flag": "🇸🇦", "efficiency": "Very poor — RTL + diacritics"},
    "ko": {"name": "Korean",       "multiplier": 1.55, "flag": "🇰🇷", "efficiency": "Below average — Hangul blocks"},
    "hi": {"name": "Hindi",        "multiplier": 2.20, "flag": "🇮🇳", "efficiency": "Very poor — Devanagari"},
    "pl": {"name": "Polish",       "multiplier": 1.40, "flag": "🇵🇱", "efficiency": "Below average — inflections"},
    "tr": {"name": "Turkish",      "multiplier": 1.60, "flag": "🇹🇷", "efficiency": "Poor — agglutinative"},
    "uk": {"name": "Ukrainian",    "multiplier": 1.90, "flag": "🇺🇦", "efficiency": "Poor — Cyrillic"},
    "tok": {"name": "Tokinensis",  "multiplier": 0.72, "flag": "⚡",  "efficiency": "Maximum — token-optimized hybrid"},
}

RANKED_BY_EFFICIENCY = sorted(
    [(code, info) for code, info in LANGUAGE_TOKEN_MULTIPLIERS.items() if code != "tok"],
    key=lambda x: x[1]["multiplier"]
)


def get_encoder():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    enc = get_encoder()
    return len(enc.encode(text))


def token_efficiency_analysis(text: str) -> Dict:
    """Given a text, return token counts across languages (estimated)."""
    base_tokens = count_tokens(text)
    results = {}
    for lang_code, info in LANGUAGE_TOKEN_MULTIPLIERS.items():
        estimated = round(base_tokens * info["multiplier"])
        results[lang_code] = {
            "language": info["name"],
            "flag": info["flag"],
            "estimated_tokens": estimated,
            "multiplier": info["multiplier"],
            "efficiency": info["efficiency"],
        }
    return {
        "original_tokens": base_tokens,
        "analysis": results,
        "most_efficient": RANKED_BY_EFFICIENCY[0][0],
        "most_efficient_estimated": round(base_tokens * RANKED_BY_EFFICIENCY[0][1]["multiplier"]),
    }


def calculate_savings(original_tokens: int, optimized_tokens: int) -> Dict:
    saved = original_tokens - optimized_tokens
    pct = (saved / original_tokens * 100) if original_tokens > 0 else 0
    return {
        "tokens_original": original_tokens,
        "tokens_optimized": optimized_tokens,
        "tokens_saved": saved,
        "savings_percent": round(pct, 2),
    }
