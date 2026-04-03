"""
TokenTranslation Skill Client
================================
Include this file in your Claude skill / MCP server for instant token savings.

USAGE (3-4 lines):
    from skill_client import TokenTranslationClient
    tt = TokenTranslationClient("http://localhost:8080", api_key="tk_your_key")
    optimized, meta = await tt.translate_in("Hola, cómo estás?")
    # ... call your LLM with `optimized` ...
    final_response = await tt.translate_out(llm_response, meta["source_lang"])

With Tokinensis (maximum savings):
    optimized, meta = await tt.translate_in(prompt, use_tokinensis=True)
    final_response = await tt.translate_out(llm_response, meta["source_lang"], was_tokinensis=True)
"""

import httpx
from typing import Tuple, Optional


class TokenTranslationClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.timeout = timeout

    async def translate_in(
        self,
        text: str,
        use_tokinensis: bool = False,
        backend: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Translate input to the most token-efficient form.
        Returns: (optimized_text, metadata_dict)
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/translate/in",
                json={"text": text, "use_tokinensis": use_tokinensis, "backend": backend},
                headers=self.headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["optimized_text"], data

    async def translate_out(
        self,
        text: str,
        target_lang: str,
        was_tokinensis: bool = False,
        backend: Optional[str] = None,
    ) -> str:
        """
        Translate model response back to user's language.
        Returns: translated_text
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/translate/out",
                json={"text": text, "target_lang": target_lang, "was_tokinensis": was_tokinensis, "backend": backend},
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()["text"]

    async def analyze(self, text: str) -> dict:
        """Analyze token costs for a text across all languages."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/translate/analyze",
                json={"text": text},
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def my_stats(self) -> dict:
        """Get your token savings statistics."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/stats/me", headers=self.headers)
            resp.raise_for_status()
            return resp.json()


# ─── Synchronous wrapper for non-async environments ─────────────────────────
import asyncio


class SyncTokenTranslationClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self._async = TokenTranslationClient(base_url, api_key, timeout)

    def translate_in(self, text: str, use_tokinensis: bool = False) -> Tuple[str, dict]:
        return asyncio.run(self._async.translate_in(text, use_tokinensis))

    def translate_out(self, text: str, target_lang: str, was_tokinensis: bool = False) -> str:
        return asyncio.run(self._async.translate_out(text, target_lang, was_tokinensis))


# ─── Example usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def example():
        tt = TokenTranslationClient("http://localhost:8080", api_key="tk_your_key_here")

        # Translate a Spanish prompt
        optimized, meta = await tt.translate_in("Explica cómo funciona el aprendizaje automático")
        print(f"Optimized: {optimized}")
        print(f"Tokens saved: {meta['tokens_saved']} ({meta['savings_percent']}%)")

        # Simulate LLM response and translate back
        fake_response = "Machine learning is a subset of AI that learns from data."
        final = await tt.translate_out(fake_response, meta["source_lang"])
        print(f"Final response (in Spanish): {final}")

    asyncio.run(example())
