"""Main translation router — routes to local or Google backend."""
import logging
from typing import Optional

from langdetect import detect as langdetect_detect, LangDetectException
from config import settings
from services import translator_local, translator_google, tokinensis, tokinensis_v2
from services.token_counter import count_tokens, calculate_savings

logger = logging.getLogger(__name__)


async def detect_language(text: str) -> str:
    try:
        code = langdetect_detect(text)
        if code and len(code) == 2:
            return code
    except LangDetectException:
        pass
    try:
        return await translator_google.detect_language(text) or "en"
    except Exception:
        return "en"


async def translate(text: str, source_lang: str, target_lang: str, backend: Optional[str] = None) -> Optional[str]:
    if source_lang == target_lang:
        return text
    chosen = backend or settings.translation_backend
    if chosen == "local":
        result = await translator_local.translate(text, source_lang, target_lang)
        if result:
            return result
        return await translator_google.translate(text, source_lang, target_lang)
    else:
        result = await translator_google.translate(text, source_lang, target_lang)
        if result:
            return result
        return await translator_local.translate(text, source_lang, target_lang)


async def process_input(
    text: str,
    use_tokinensis: bool = False,
    tokinensis_version: int = 1,
    backend: Optional[str] = None,
) -> dict:
    source_lang = await detect_language(text)
    target_lang = settings.efficient_language
    tokens_original = count_tokens(text)

    if source_lang != target_lang:
        translated = await translate(text, source_lang, target_lang, backend)
        if not translated:
            translated = text
    else:
        translated = text

    tokens_translated = count_tokens(translated)
    final_text = translated
    tokens_tokinensis = None

    if use_tokinensis:
        if tokinensis_version == 2:
            tok_text, _, tok_count, _, _ = tokinensis_v2.encode(translated)
        else:
            tok_text, _, tok_count = tokinensis.encode(translated)
        final_text = tok_text
        tokens_tokinensis = tok_count

    tokens_final = count_tokens(final_text)
    savings = calculate_savings(tokens_original, tokens_final)

    return {
        "original_text": text,
        "optimized_text": final_text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "tokens_original": tokens_original,
        "tokens_translated": tokens_translated,
        "tokens_tokinensis": tokens_tokinensis,
        "tokens_final": tokens_final,
        "savings": savings,
        "backend_used": backend or settings.translation_backend,
        "tokinensis_applied": use_tokinensis,
    }


async def process_output(
    text: str,
    target_lang: str,
    was_tokinensis: bool = False,
    tokinensis_version: int = 1,
    backend: Optional[str] = None,
) -> str:
    working_text = text
    if was_tokinensis:
        if tokinensis_version == 2:
            working_text = tokinensis_v2.decode(working_text)
        else:
            working_text = tokinensis.decode(working_text)

    source_lang = settings.efficient_language
    if target_lang != source_lang:
        translated = await translate(working_text, source_lang, target_lang, backend)
        return translated or working_text
    return working_text
