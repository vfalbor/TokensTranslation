from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from auth import (
    authenticate_user, create_access_token, generate_api_key,
    get_current_user, get_password_hash,
    get_user_by_email, get_user_by_username,
    get_user_by_google_id, get_user_by_verification_token,
)
from config import settings
from database import User, get_db
from services.email_service import (
    generate_verification_token, send_verification_email, send_api_key_email,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    api_key: str
    email_verified: bool
    total_requests: int
    total_tokens_original: float
    total_tokens_optimized: float

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    api_key: str
    user: UserResponse


# ── Register ───────────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if await get_user_by_email(db, body.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await get_user_by_username(db, body.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    verification_token = generate_verification_token()
    user = User(
        email=body.email,
        username=body.username,
        hashed_password=get_password_hash(body.password),
        api_key=generate_api_key(),
        email_verified=not settings.require_email_verification,
        verification_token=verification_token if settings.require_email_verification else None,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Send verification email (non-blocking; fails gracefully)
    if settings.require_email_verification:
        await send_verification_email(user.email, user.username, verification_token)

    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token, api_key=user.api_key, user=UserResponse.model_validate(user))


# ── Login ──────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token, api_key=user.api_key, user=UserResponse.model_validate(user))


# ── Email verification ─────────────────────────────────────────────────────────
@router.get("/verify/{token}")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_verification_token(db, token)
    if not user:
        raise HTTPException(status_code=404, detail="Invalid or expired verification token")
    if user.email_verified:
        return {"message": "Email already verified", "status": "already_verified"}

    user.email_verified = True
    user.verification_token = None
    await db.commit()

    # Send API key reminder
    await send_api_key_email(user.email, user.username, user.api_key)

    # Redirect to dashboard
    return RedirectResponse(url=f"{settings.frontend_url}/dashboard.html?verified=1")


@router.post("/resend-verification")
async def resend_verification(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.email_verified:
        return {"message": "Email already verified"}
    token = generate_verification_token()
    current_user.verification_token = token
    await db.commit()
    await send_verification_email(current_user.email, current_user.username, token)
    return {"message": "Verification email sent"}


# ── Google OAuth ───────────────────────────────────────────────────────────────
@router.get("/google")
async def google_login():
    """Redirect user to Google OAuth consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured. Add GOOGLE_CLIENT_ID to .env")

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
    }
    import urllib.parse
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=url)


@router.get("/google/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Handle Google OAuth callback, create/find user, return token."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    import httpx

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Google token exchange failed")
        token_data = token_resp.json()

        # Get user info
        info_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        if info_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get Google user info")
        google_user = info_resp.json()

    google_id = google_user.get("id")
    email = google_user.get("email")
    name = google_user.get("name", "").replace(" ", "_").lower()[:20] or "user"

    # Find or create user
    user = await get_user_by_google_id(db, google_id)
    if not user:
        user = await get_user_by_email(db, email)
        if user:
            # Link Google ID to existing account
            user.google_id = google_id
            user.email_verified = True
        else:
            # Create new account
            username = name
            base = username
            counter = 1
            while await get_user_by_username(db, username):
                username = f"{base}{counter}"
                counter += 1

            user = User(
                email=email,
                username=username,
                hashed_password=None,
                api_key=generate_api_key(),
                google_id=google_id,
                email_verified=True,
            )
            db.add(user)
    await db.commit()
    await db.refresh(user)

    access_token = create_access_token({"sub": user.email})
    # Redirect to frontend with token
    return RedirectResponse(
        url=f"{settings.frontend_url}/dashboard.html?token={access_token}&key={user.api_key}"
    )


# ── Me / key rotation ──────────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.post("/rotate-key")
async def rotate_api_key(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    current_user.api_key = generate_api_key()
    await db.commit()
    await db.refresh(current_user)
    return {"api_key": current_user.api_key, "message": "API key rotated successfully"}
