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
from typing import Callable, Optional, Awaitable, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.services.google import GoogleTTSService
from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner


# Voice configurations for each bot
GOOGLE_TTS_VOICES = {
    "Alice": "en-US-Studio-O",  # Female
    "Bob": "en-US-Studio-M",    # Male
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
        # Build the full prompt
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

    Instead of continuous streaming, this gives full control:
    - Generate text response
    - Convert to speech
    - Play audio
    - Wait for completion
    - Next turn
    """

    def __init__(self):
        self.llm = TextLLMService()
        self._google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def create_bot_config(self, name: str, system_prompt: str) -> BotConfig:
        """Create a bot configuration."""
        voice_id = GOOGLE_TTS_VOICES.get(name, "en-US-Studio-O")
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

    def create_tts_service(self, bot: BotConfig) -> GoogleTTSService:
        """Create a TTS service for this bot."""
        return GoogleTTSService(
            api_key=self._google_api_key,
            voice_id=bot.voice_id,
        )

    async def speak_text(
        self,
        transport: DailyTransport,
        bot: BotConfig,
        text: str,
    ) -> None:
        """
        Speak the given text through the Daily transport.

        Creates a simple pipeline: TTS → Transport output
        Waits for audio playback to complete.
        """
        tts = self.create_tts_service(bot)

        # Create a simple output-only pipeline
        pipeline = Pipeline([
            tts,
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
            ),
        )

        # Queue the text to speak
        await task.queue_frames([
            TTSSpeakFrame(text=text),
            EndFrame(),
        ])

        # Run until complete
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

        print(f"[TTS] {bot.name} finished speaking")


class SimpleDailyTransport:
    """
    Simplified Daily transport for audio output only.
    """

    @staticmethod
    def create(room_url: str, token: str, bot_name: str) -> DailyTransport:
        """Create a Daily transport configured for output only."""
        return DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_in_enabled=False,   # No input needed
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_enabled=False,
                transcription_enabled=False,
            ),
        )
