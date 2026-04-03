from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    app_name: str = "TokenTranslation"
    app_version: str = "1.1.0"
    debug: bool = False
    frontend_url: str = "https://tokenstree.eu"

    # Security
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_openssl_rand_hex_32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 30  # 30 days

    # Database
    database_url: str = "sqlite+aiosqlite:///./tokentranslation.db"

    # Translation backend: "local" | "google"
    translation_backend: Literal["local", "google"] = "google"
    google_translate_api_key: str = ""
    efficient_language: str = "en"

    # Rate limiting
    rate_limit: str = "100/minute"
    cors_origins: list[str] = ["*"]

    # ── Email (for registration confirmation) ─────────────────────────────
    # Set MAIL_USERNAME + MAIL_PASSWORD to enable.
    # Works with Gmail (use App Password), Mailgun, SendGrid SMTP, etc.
    mail_username: str = ""
    mail_password: str = ""
    mail_from: str = "noreply@tokenstree.eu"
    mail_from_name: str = "TokenTranslation"
    mail_server: str = "smtp.gmail.com"
    mail_port: int = 587
    mail_tls: bool = True
    mail_ssl: bool = False
    # If True, users must confirm email before using the API.
    # Set False to disable email verification (useful during dev).
    require_email_verification: bool = False

    # ── Google OAuth ───────────────────────────────────────────────────────
    # Create credentials at: https://console.cloud.google.com/apis/credentials
    # Authorized redirect URI: https://tokenstree.eu/auth/google/callback
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "https://tokenstree.eu/auth/google/callback"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
