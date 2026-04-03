from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime, timezone, timedelta

from auth import get_current_user
from database import User, TranslationLog, get_db
from services.translator_local import get_model_status
from config import settings

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/me")
async def my_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return token savings statistics for the authenticated user."""
    total_saved = current_user.total_tokens_original - current_user.total_tokens_optimized
    pct = (
        (total_saved / current_user.total_tokens_original * 100)
        if current_user.total_tokens_original > 0
        else 0
    )

    # Recent activity (last 20 translations)
    result = await db.execute(
        select(TranslationLog)
        .where(TranslationLog.user_id == current_user.id)
        .order_by(desc(TranslationLog.created_at))
        .limit(20)
    )
    logs = result.scalars().all()

    # Per-language breakdown
    lang_result = await db.execute(
        select(
            TranslationLog.source_lang,
            func.count(TranslationLog.id).label("count"),
            func.sum(TranslationLog.tokens_saved).label("total_saved"),
        )
        .where(TranslationLog.user_id == current_user.id)
        .group_by(TranslationLog.source_lang)
        .order_by(desc("total_saved"))
    )
    lang_stats = [
        {"lang": row.source_lang, "requests": row.count, "tokens_saved": row.total_saved or 0}
        for row in lang_result.all()
    ]

    return {
        "user": {
            "username": current_user.username,
            "member_since": current_user.created_at.isoformat(),
        },
        "totals": {
            "requests": current_user.total_requests,
            "tokens_original": current_user.total_tokens_original,
            "tokens_optimized": current_user.total_tokens_optimized,
            "tokens_saved": total_saved,
            "savings_percent": round(pct, 2),
        },
        "recent_activity": [
            {
                "id": log.id,
                "source_lang": log.source_lang,
                "target_lang": log.target_lang,
                "tokens_original": log.tokens_original,
                "tokens_optimized": log.tokens_optimized,
                "tokens_saved": log.tokens_saved,
                "backend": log.backend_used,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "by_language": lang_stats,
    }


@router.get("/global")
async def global_stats(db: AsyncSession = Depends(get_db)):
    """Public global statistics — no auth required."""
    result = await db.execute(
        select(
            func.count(TranslationLog.id).label("total_requests"),
            func.sum(TranslationLog.tokens_original).label("total_original"),
            func.sum(TranslationLog.tokens_optimized).label("total_optimized"),
            func.sum(TranslationLog.tokens_saved).label("total_saved"),
        )
    )
    row = result.one()
    total_original = row.total_original or 0
    total_saved = row.total_saved or 0
    pct = (total_saved / total_original * 100) if total_original > 0 else 0

    user_count = await db.execute(select(func.count(User.id)))
    users = user_count.scalar() or 0

    return {
        "total_users": users,
        "total_requests": row.total_requests or 0,
        "total_tokens_processed": total_original,
        "total_tokens_saved": total_saved,
        "average_savings_percent": round(pct, 2),
    }


@router.get("/system")
async def system_status():
    """Return backend configuration and local model status."""
    return {
        "translation_backend": settings.translation_backend,
        "efficient_language": settings.efficient_language,
        "google_api_configured": bool(settings.google_translate_api_key),
        "local_models": get_model_status(),
    }
