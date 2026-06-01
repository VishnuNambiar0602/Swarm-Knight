"""Open-source LLM provider abstraction layer."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import httpx

from .models import ProviderType, SwarmParticipant

logger = logging.getLogger(__name__)


class OpenSourceProvider(ABC):
    """Abstract base class for open-source LLM providers."""

    def __init__(self, participant: SwarmParticipant):
        self.participant = participant
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(timeout=120.0)
        return self.client

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion from the LLM."""
        pass

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models from this provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable and healthy."""
        pass

    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """Generate with exponential backoff retry."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.generate(prompt, system_prompt, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0
                    logger.warning(
                        f"Provider {self.participant.name} attempt {attempt + 1} "
                        f"failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
        raise last_error


class OpenRouterProvider(OpenSourceProvider):
    """OpenRouter provider for free models."""

    BASE_URL = "https://openrouter.ai/api/v1"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.participant.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openswarm.com",
            "X-Title": "OpenSwarm",
        }

        payload = {
            "model": self.participant.model,
            "messages": messages,
            "temperature": temperature or self.participant.temperature,
            "max_tokens": max_tokens or self.participant.max_tokens,
        }

        response = await client.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.participant.api_key}",
        }

        response = await client.get(
            f"{self.BASE_URL}/models",
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        return [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length", 0),
                "pricing": m.get("pricing", {}),
            }
            for m in models
            if ":free" in m["id"] or m.get("pricing", {}).get("prompt") == "0"
        ]

    async def health_check(self) -> bool:
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception as e:
            logger.error(f"OpenRouter health check failed: {e}")
            return False


class OllamaProvider(OpenSourceProvider):
    """Ollama provider for local LLM inference."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, participant: SwarmParticipant):
        super().__init__(participant)
        self.base_url = participant.base_url or self.DEFAULT_BASE_URL

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = await self._get_client()

        payload = {
            "model": self.participant.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.participant.temperature,
                "num_predict": max_tokens or self.participant.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = await client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

    async def list_models(self) -> list[dict]:
        client = await self._get_client()

        try:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            return [
                {
                    "id": m["name"],
                    "name": m.get("name", m["name"]),
                    "size": m.get("size", 0),
                    "modified": m.get("modified_at", ""),
                }
                for m in models
            ]
        except Exception as e:
            logger.error(f"Ollama list models failed: {e}")
            return []

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class TogetherProvider(OpenSourceProvider):
    """Together AI provider."""

    BASE_URL = "https://api.together.xyz/v1"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.participant.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.participant.model,
            "messages": messages,
            "temperature": temperature or self.participant.temperature,
            "max_tokens": max_tokens or self.participant.max_tokens,
        }

        response = await client.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.participant.api_key}",
        }

        response = await client.get(
            f"{self.BASE_URL}/models",
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        return [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
            }
            for m in data
        ]

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            headers = {"Authorization": f"Bearer {self.participant.api_key}"}
            response = await client.get(f"{self.BASE_URL}/models", headers=headers)
            return response.status_code == 200
        except Exception:
            return False


class GroqProvider(OpenSourceProvider):
    """Groq provider for fast inference."""

    BASE_URL = "https://api.groq.com/openai/v1"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.participant.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.participant.model,
            "messages": messages,
            "temperature": temperature or self.participant.temperature,
            "max_tokens": max_tokens or self.participant.max_tokens,
        }

        response = await client.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        headers = {"Authorization": f"Bearer {self.participant.api_key}"}

        response = await client.get(f"{self.BASE_URL}/models", headers=headers)
        response.raise_for_status()

        data = response.json()
        return [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
            }
            for m in data.get("data", [])
        ]

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            headers = {"Authorization": f"Bearer {self.participant.api_key}"}
            response = await client.get(f"{self.BASE_URL}/models", headers=headers)
            return response.status_code == 200
        except Exception:
            return False


def create_provider(participant: SwarmParticipant) -> OpenSourceProvider:
    """Factory function to create the appropriate provider."""
    providers = {
        ProviderType.OPENROUTER: OpenRouterProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.TOGETHER: TogetherProvider,
        ProviderType.GROQ: GroqProvider,
    }

    provider_class = providers.get(participant.provider)
    if provider_class is None:
        raise ValueError(f"Unsupported provider: {participant.provider}")

    return provider_class(participant)
