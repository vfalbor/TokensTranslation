from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from datetime import datetime, timezone
from config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # Null for Google-OAuth users
    api_key = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    google_id = Column(String, unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Accumulated stats
    total_tokens_original = Column(Float, default=0.0)
    total_tokens_optimized = Column(Float, default=0.0)
    total_requests = Column(Integer, default=0)


class TranslationLog(Base):
    __tablename__ = "translation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    source_lang = Column(String(10), nullable=False)
    target_lang = Column(String(10), nullable=False)
    original_text_len = Column(Integer)
    translated_text_len = Column(Integer)
    tokens_original = Column(Float)
    tokens_optimized = Column(Float)
    tokens_saved = Column(Float)
    backend_used = Column(String(20))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
