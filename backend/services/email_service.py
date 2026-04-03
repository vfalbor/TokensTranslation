"""
Email service for TokenTranslation.
Sends email verification links after registration.

Backends supported (configure via .env):
  - Gmail (MAIL_SERVER=smtp.gmail.com, use App Password)
  - Mailgun SMTP
  - SendGrid SMTP
  - Any standard SMTP server

If MAIL_USERNAME is empty, email sending is disabled and the
verification token is returned in the API response for development.
"""
import logging
import secrets
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_mail_client = None


def _get_mail():
    global _mail_client
    if _mail_client is None and settings.mail_username:
        try:
            from fastapi_mail import FastMail, ConnectionConfig
            conf = ConnectionConfig(
                MAIL_USERNAME=settings.mail_username,
                MAIL_PASSWORD=settings.mail_password,
                MAIL_FROM=settings.mail_from,
                MAIL_FROM_NAME=settings.mail_from_name,
                MAIL_PORT=settings.mail_port,
                MAIL_SERVER=settings.mail_server,
                MAIL_STARTTLS=settings.mail_tls,
                MAIL_SSL_TLS=settings.mail_ssl,
                USE_CREDENTIALS=True,
                VALIDATE_CERTS=True,
            )
            _mail_client = FastMail(conf)
        except Exception as e:
            logger.error(f"Failed to initialize mail client: {e}")
    return _mail_client


def generate_verification_token() -> str:
    return secrets.token_urlsafe(32)


async def send_verification_email(email: str, username: str, token: str) -> bool:
    """Send an email verification link. Returns True if sent, False if skipped."""
    verify_url = f"{settings.frontend_url}/verify?token={token}"
    mail = _get_mail()

    if not mail:
        logger.info(f"[DEV] Email verification disabled. Token for {email}: {token}")
        logger.info(f"[DEV] Verify URL: {verify_url}")
        return False

    html = f"""
    <div style="font-family: 'DM Sans', sans-serif; max-width: 560px; margin: 0 auto; padding: 2rem;">
      <div style="margin-bottom: 2rem;">
        <span style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">⚡ TokenTranslation</span>
      </div>
      <h2 style="color: #1e2130; font-size: 1.4rem; margin-bottom: 1rem;">Verify your email address</h2>
      <p style="color: #4a5168; line-height: 1.7; margin-bottom: 1.5rem;">
        Hi <strong>{username}</strong>, thanks for registering!<br>
        Click the button below to confirm your email and activate your account.
      </p>
      <a href="{verify_url}"
         style="display: inline-block; background: #2563eb; color: white; padding: .85rem 2rem;
                border-radius: 8px; font-weight: 700; text-decoration: none; font-size: .95rem;">
        Verify Email Address
      </a>
      <p style="color: #8b92a8; font-size: .82rem; margin-top: 2rem;">
        This link expires in 24 hours. If you didn't create this account, ignore this email.<br>
        <a href="{verify_url}" style="color: #8b92a8;">{verify_url}</a>
      </p>
    </div>
    """

    try:
        from fastapi_mail import MessageSchema, MessageType
        message = MessageSchema(
            subject="Verify your TokenTranslation account",
            recipients=[email],
            body=html,
            subtype=MessageType.html,
        )
        await mail.send_message(message)
        logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email to {email}: {e}")
        return False


async def send_api_key_email(email: str, username: str, api_key: str) -> bool:
    """Send API key reminder after verification."""
    mail = _get_mail()
    if not mail:
        return False

    html = f"""
    <div style="font-family: 'DM Sans', sans-serif; max-width: 560px; margin: 0 auto; padding: 2rem;">
      <div style="margin-bottom: 2rem;">
        <span style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">⚡ TokenTranslation</span>
      </div>
      <h2 style="color: #1e2130;">Your account is ready, {username}!</h2>
      <p style="color: #4a5168; line-height: 1.7; margin-bottom: 1.5rem;">
        Your email has been verified. Here is your API key:
      </p>
      <div style="background: #eff4ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
        <div style="font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #2563eb; margin-bottom: .4rem;">API KEY</div>
        <code style="font-family: 'Courier New', monospace; font-size: .9rem; color: #1e2130;">{api_key}</code>
      </div>
      <p style="color: #8b92a8; font-size: .82rem;">Keep this key private. You can always view or rotate it in your Dashboard.</p>
      <a href="{settings.frontend_url}/dashboard.html"
         style="display: inline-block; background: #2563eb; color: white; padding: .75rem 1.75rem;
                border-radius: 8px; font-weight: 700; text-decoration: none; font-size: .9rem; margin-top: .75rem;">
        Open Dashboard →
      </a>
    </div>
    """

    try:
        from fastapi_mail import MessageSchema, MessageType
        message = MessageSchema(
            subject="Your TokenTranslation API key",
            recipients=[email],
            body=html,
            subtype=MessageType.html,
        )
        await mail.send_message(message)
        return True
    except Exception as e:
        logger.error(f"Failed to send API key email: {e}")
        return False
