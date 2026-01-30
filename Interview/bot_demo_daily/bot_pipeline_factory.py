"""
=============================================================================
BOT PIPELINE FACTORY - Sequential Turn-Based Bot Conversations
=============================================================================

This module creates a simple turn-based conversation system where:
1. Alice generates text → TTS → plays audio → waits for completion
2. Bob generates text → TTS → plays audio → waits for completion
3. Repeat until conversation ends

No continuous streaming - full control over turn-taking.
"""

import os
import asyncio
from typing import List, Dict
from dataclasses import dataclass, field

import google.generativeai as genai

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.services.elevenlabs import ElevenLabsTTSService


# ElevenLabs voice IDs for each bot
ELEVENLABS_VOICES = {
    "Alice": "21m00Tcm4TlvDq8ikWAM",  # Rachel - female
    "Bob": "ErXwobaYiN019PkySvjV",    # Antoni - male
}


@dataclass
class BotConfig:
    """Configuration for a single bot."""
    name: str
    system_prompt: str
    voice_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


class TextLLMService:
    """Simple wrapper for Gemini text generation."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def generate(self, system_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a text response given system prompt and conversation history."""
        full_prompt = f"{system_prompt}\n\nConversation so far:\n"
        for msg in conversation_history:
            full_prompt += f"{msg['speaker']}: {msg['text']}\n"
        full_prompt += "\nRespond with your next line only (1-2 sentences). Do not include your name prefix."

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"[LLM] Error generating response: {e}")
            return "I'm sorry, I'm having trouble responding right now."


class TurnBasedBotFactory:
    """
    Factory for creating turn-based bot conversations.
    """

    def __init__(self):
        self.llm = TextLLMService()
        self._elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self._elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

    def create_bot_config(self, name: str, system_prompt: str) -> BotConfig:
        """Create a bot configuration."""
        voice_id = ELEVENLABS_VOICES.get(name, ELEVENLABS_VOICES["Alice"])
        return BotConfig(
            name=name,
            system_prompt=system_prompt,
            voice_id=voice_id,
            conversation_history=[]
        )

    async def generate_response(self, bot: BotConfig) -> str:
        """Generate a text response for this bot."""
        text = await self.llm.generate(bot.system_prompt, bot.conversation_history)
        print(f"[LLM] {bot.name} generated: {text[:100]}...")
        return text

    def create_tts_service(self, bot: BotConfig) -> ElevenLabsTTSService:
        """Create an ElevenLabs TTS service for this bot."""
        return ElevenLabsTTSService(
            api_key=self._elevenlabs_api_key,
            voice_id=bot.voice_id,
        )
