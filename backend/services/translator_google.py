"""
Google Translate backend using deep-translator (reliable, actively maintained).
Two modes:
  1. Free via deep-translator GoogleTranslator — no API key needed.
  2. Official via Google Cloud Translation API — requires GOOGLE_TRANSLATE_API_KEY.

The system auto-selects: if an API key is configured, uses the official API.
"""
import asyncio
import logging
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

_OFFICIAL_BASE = "https://translation.googleapis.com/language/translate/v2"


async def _translate_official(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """Use Google Cloud Translation API (requires API key)."""
    params = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
        "key": settings.google_translate_api_key,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_OFFICIAL_BASE, json=params)
        resp.raise_for_status()
        data = resp.json()
        return data["data"]["translations"][0]["translatedText"]


async def _translate_free(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """Use deep-translator's GoogleTranslator (free, no API key)."""
    loop = asyncio.get_event_loop()
    try:
        from deep_translator import GoogleTranslator
        result = await loop.run_in_executor(
            None,
            lambda: GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        )
        return result
    except Exception as e:
        logger.error(f"deep-translator error ({source_lang}→{target_lang}): {e}")
        return None


async def translate(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """Translate text using Google Translate (official API or free deep-translator)."""
    if source_lang == target_lang:
        return text
    try:
        if settings.google_translate_api_key:
            return await _translate_official(text, source_lang, target_lang)
        else:
            return await _translate_free(text, source_lang, target_lang)
    except Exception as e:
        logger.error(f"Google Translate error ({source_lang}→{target_lang}): {e}")
        return None


async def detect_language(text: str) -> str:
    """Detect language using deep-translator."""
    loop = asyncio.get_event_loop()
    try:
        if settings.google_translate_api_key:
            params = {"q": text, "key": settings.google_translate_api_key}
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(f"{_OFFICIAL_BASE}/detect", json=params)
                resp.raise_for_status()
                data = resp.json()
                return data["data"]["detections"][0][0]["language"]
        else:
            from deep_translator import GoogleTranslator
            # Detect by translating with source='auto' and checking metadata
            translator = GoogleTranslator(source="auto", target="en")
            await loop.run_in_executor(None, lambda: translator.translate(text[:200]))
            detected = getattr(translator, '_source', 'en') or 'en'
            return detected
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return "en"
