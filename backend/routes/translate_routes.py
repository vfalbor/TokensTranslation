from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from config import settings
from database import User, TranslationLog, get_db
from services import translation_router
from services import tokinensis, tokinensis_v2
from services.token_counter import LANGUAGE_TOKEN_MULTIPLIERS, RANKED_BY_EFFICIENCY, token_efficiency_analysis

router = APIRouter(prefix="/translate", tags=["Translation"])


class TranslateInRequest(BaseModel):
    text: str
    use_tokinensis: bool = False
    tokinensis_version: int = 1   # 1 or 2
    backend: Optional[str] = None


class TranslateInResponse(BaseModel):
    original_text: str
    optimized_text: str
    source_lang: str
    target_lang: str
    tokens_original: int
    tokens_final: int
    tokens_saved: int
    savings_percent: float
    backend_used: str
    tokinensis_applied: bool
    tokinensis_version: Optional[int] = None


class TranslateOutRequest(BaseModel):
    text: str
    target_lang: str
    was_tokinensis: bool = False
    tokinensis_version: int = 1
    backend: Optional[str] = None


class TextRequest(BaseModel):
    text: str


class ConceptCompareRequest(BaseModel):
    forms: dict  # {"en": "superhuman", "es": "superhumano", "zh": "超人", "tok_v2": "gra-hom"}


@router.post("/in", response_model=TranslateInResponse)
async def translate_in(
    body: TranslateInRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await translation_router.process_input(
        body.text,
        use_tokinensis=body.use_tokinensis,
        tokinensis_version=body.tokinensis_version,
        backend=body.backend,
    )

    current_user.total_requests += 1
    current_user.total_tokens_original += result["tokens_original"]
    current_user.total_tokens_optimized += result["tokens_final"]

    log = TranslationLog(
        user_id=current_user.id,
        source_lang=result["source_lang"],
        target_lang=result["target_lang"],
        original_text_len=len(body.text),
        translated_text_len=len(result["optimized_text"]),
        tokens_original=result["tokens_original"],
        tokens_optimized=result["tokens_final"],
        tokens_saved=result["savings"]["tokens_saved"],
        backend_used=result["backend_used"],
    )
    db.add(log)
    await db.commit()

    return TranslateInResponse(
        original_text=result["original_text"],
        optimized_text=result["optimized_text"],
        source_lang=result["source_lang"],
        target_lang=result["target_lang"],
        tokens_original=result["tokens_original"],
        tokens_final=result["tokens_final"],
        tokens_saved=result["savings"]["tokens_saved"],
        savings_percent=result["savings"]["savings_percent"],
        backend_used=result["backend_used"],
        tokinensis_applied=result["tokinensis_applied"],
        tokinensis_version=body.tokinensis_version if body.use_tokinensis else None,
    )


@router.post("/out")
async def translate_out(
    body: TranslateOutRequest,
    current_user: User = Depends(get_current_user),
):
    text = await translation_router.process_output(
        body.text,
        target_lang=body.target_lang,
        was_tokinensis=body.was_tokinensis,
        tokinensis_version=body.tokinensis_version,
        backend=body.backend,
    )
    return {"text": text}


@router.post("/analyze")
async def analyze_text(body: TextRequest, current_user: User = Depends(get_current_user)):
    return token_efficiency_analysis(body.text)


@router.get("/languages")
async def get_languages():
    return {
        "languages": LANGUAGE_TOKEN_MULTIPLIERS,
        "ranked_by_efficiency": [{"code": c, **i} for c, i in RANKED_BY_EFFICIENCY],
        "most_efficient": RANKED_BY_EFFICIENCY[0][0],
    }


# ── Tokinensis v1 ────────────────────────────────────────────────────────────
@router.post("/tokinensis/encode")
async def tok_v1_encode(body: TextRequest, current_user: User = Depends(get_current_user)):
    encoded, orig, opt = tokinensis.encode(body.text)
    return {
        "version": 1,
        "original": body.text,
        "encoded": encoded,
        "tokens_original": orig,
        "tokens_optimized": opt,
        "tokens_saved": orig - opt,
        "savings_pct": round((orig - opt) / orig * 100, 1) if orig > 0 else 0,
    }


@router.post("/tokinensis/decode")
async def tok_v1_decode(body: TextRequest, current_user: User = Depends(get_current_user)):
    return {"decoded": tokinensis.decode(body.text)}


@router.get("/tokinensis/samples")
async def tok_v1_samples(current_user: User = Depends(get_current_user)):
    return tokinensis.get_sample_comparisons()


# ── Tokinensis v2 ────────────────────────────────────────────────────────────
@router.post("/tokinensis/v2/encode")
async def tok_v2_encode(body: TextRequest, current_user: User = Depends(get_current_user)):
    encoded, orig, opt, glossary, detected_lang = tokinensis_v2.encode(body.text)
    return {
        "version": 2,
        "original": body.text,
        "encoded": encoded,
        "tokens_original": orig,
        "tokens_optimized": opt,
        "tokens_saved": orig - opt,
        "savings_pct": round((orig - opt) / orig * 100, 1) if orig > 0 else 0,
        "glossary": glossary,
        "detected_lang": detected_lang,
    }


@router.post("/tokinensis/v2/decode")
async def tok_v2_decode(body: TextRequest, current_user: User = Depends(get_current_user)):
    return {"decoded": tokinensis_v2.decode(body.text)}


@router.get("/tokinensis/v2/samples")
async def tok_v2_samples(current_user: User = Depends(get_current_user)):
    return tokinensis_v2.get_sample_comparisons()


@router.get("/tokinensis/v2/vocabulary")
async def tok_v2_vocabulary(current_user: User = Depends(get_current_user)):
    return {
        "vocab": tokinensis_v2.get_vocabulary_size(),
        "roots": {
            root: {lang: words[:3] for lang, words in langs.items()}
            for root, langs in list(tokinensis_v2.ROOTS.items())[:40]  # first 40
        },
    }


@router.post("/tokinensis/v2/compare-optimal")
async def tok_v2_compare_optimal(body: ConceptCompareRequest, current_user: User = Depends(get_current_user)):
    """Find which language/form uses fewest tokens for a given concept."""
    return tokinensis_v2.compare_optimal_tokens(body.forms)


@router.get("/tokinensis/v2/samples/public")
async def tok_v2_samples_public():
    """Public endpoint — no auth required (for landing page demo)."""
    return tokinensis_v2.get_sample_comparisons()
