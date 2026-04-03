"""
Local translation backend using argostranslate.
Runs entirely on-device — no external API calls.
Models are ~300-400 MB each (CTranslate2 format).

VPS Requirements (8GB RAM, 4 cores):
  - Each loaded model uses ~300-500 MB RAM
  - Safely run 4-6 language pairs simultaneously
  - Translation speed: ~50-200 tokens/sec on CPU
  - Recommended: preload en↔{es,fr,de,pt,it,zh} at startup

Install models by running: python install_models.py
"""
import asyncio
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import — argostranslate is optional
_argos_available = False
try:
    import argostranslate.package
    import argostranslate.translate
    _argos_available = True
except ImportError:
    logger.warning("argostranslate not installed. Local translation unavailable.")


SUPPORTED_PAIRS = {
    # source → target pairs available via argostranslate
    ("es", "en"), ("en", "es"),
    ("fr", "en"), ("en", "fr"),
    ("de", "en"), ("en", "de"),
    ("pt", "en"), ("en", "pt"),
    ("it", "en"), ("en", "it"),
    ("nl", "en"), ("en", "nl"),
    ("ru", "en"), ("en", "ru"),
    ("ar", "en"), ("en", "ar"),
    ("zh", "en"), ("en", "zh"),
    ("ja", "en"), ("en", "ja"),
    ("ko", "en"), ("en", "ko"),
    ("tr", "en"), ("en", "tr"),
    ("pl", "en"), ("en", "pl"),
    ("uk", "en"), ("en", "uk"),
    ("hi", "en"), ("en", "hi"),
}

_translation_cache: dict = {}


def is_available() -> bool:
    return _argos_available


def get_installed_languages() -> list:
    if not _argos_available:
        return []
    try:
        return argostranslate.translate.get_installed_languages()
    except Exception as e:
        logger.error(f"Error getting installed languages: {e}")
        return []


def _find_translation(from_code: str, to_code: str):
    """Find a translation object for a language pair."""
    cache_key = f"{from_code}→{to_code}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    installed = get_installed_languages()
    from_lang = next((l for l in installed if l.code == from_code), None)
    to_lang = next((l for l in installed if l.code == to_code), None)

    if not from_lang or not to_lang:
        return None

    translation = from_lang.get_translation(to_lang)
    _translation_cache[cache_key] = translation
    return translation


async def translate(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """Translate text using local argostranslate model."""
    if not _argos_available:
        raise RuntimeError("argostranslate is not available. Install with: pip install argostranslate")

    if source_lang == target_lang:
        return text

    if (source_lang, target_lang) not in SUPPORTED_PAIRS:
        # Try via English as pivot
        if source_lang != "en" and target_lang != "en":
            intermediate = await translate(text, source_lang, "en")
            if intermediate:
                return await translate(intermediate, "en", target_lang)
        return None

    loop = asyncio.get_event_loop()
    try:
        translation = await loop.run_in_executor(
            None, _find_translation, source_lang, target_lang
        )
        if not translation:
            logger.warning(f"No model found for {source_lang}→{target_lang}. Is the model installed?")
            return None

        translated = await loop.run_in_executor(None, translation.translate, text)
        return translated
    except Exception as e:
        logger.error(f"Local translation error ({source_lang}→{target_lang}): {e}")
        return None


def install_model(from_code: str, to_code: str) -> bool:
    """Download and install a translation model."""
    if not _argos_available:
        return False
    try:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None
        )
        if pkg:
            argostranslate.package.install_from_path(pkg.download())
            logger.info(f"Installed model: {from_code}→{to_code}")
            return True
        logger.warning(f"Model not found: {from_code}→{to_code}")
        return False
    except Exception as e:
        logger.error(f"Failed to install model {from_code}→{to_code}: {e}")
        return False


def get_model_status() -> dict:
    """Return which language pairs are ready to use."""
    if not _argos_available:
        return {"available": False, "installed_pairs": [], "message": "argostranslate not installed"}

    installed = get_installed_languages()
    installed_codes = {l.code for l in installed}
    ready_pairs = []
    missing_pairs = []

    for (src, tgt) in SUPPORTED_PAIRS:
        if src in installed_codes and tgt in installed_codes:
            from_lang = next((l for l in installed if l.code == src), None)
            to_lang = next((l for l in installed if l.code == tgt), None)
            if from_lang and to_lang and from_lang.get_translation(to_lang):
                ready_pairs.append(f"{src}→{tgt}")
            else:
                missing_pairs.append(f"{src}→{tgt}")
        else:
            missing_pairs.append(f"{src}→{tgt}")

    return {
        "available": True,
        "installed_pairs": ready_pairs,
        "missing_pairs": missing_pairs,
        "total_ready": len(ready_pairs),
    }
