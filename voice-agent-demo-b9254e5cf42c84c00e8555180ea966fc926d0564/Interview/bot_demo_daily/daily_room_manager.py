"""
=============================================================================
DAILY ROOM MANAGER - Daily.co REST API Wrapper
=============================================================================

This module handles all Daily.co REST API operations:
- Creating rooms for bot conversations
- Generating participant tokens (separate tokens for Alice and Bob)
- Cleaning up rooms when conversations end

AUTHENTICATION:
---------------
All requests require the DAILY_API_KEY environment variable.
Get your API key from https://dashboard.daily.co/
"""

import os
import aiohttp
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class RoomInfo:
    """Information about a Daily room and participant tokens."""
    room_url: str
    room_name: str
    alice_token: str
    bob_token: str


class DailyRoomManager:
    """
    Manages Daily.co room lifecycle and token generation.
    """

    DAILY_API_URL = "https://api.daily.co/v1"

    def __init__(self):
        """Initialize with API key from environment."""
        self.api_key = os.getenv("DAILY_API_KEY")
        if not self.api_key:
            raise ValueError("DAILY_API_KEY environment variable not set")

    async def create_room(self, room_name: Optional[str] = None) -> RoomInfo:
        """
        Create a Daily room and generate tokens for Alice and Bob.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            # Create room
            room_config = {
                "properties": {
                    "enable_chat": False,
                    "enable_screenshare": False,
                    "enable_recording": False,
                    "start_video_off": True,
                    "start_audio_off": False,
                }
            }
            if room_name:
                room_config["name"] = room_name

            async with session.post(
                f"{self.DAILY_API_URL}/rooms",
                headers=headers,
                json=room_config
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to create room: {response.status} - {error_text}")

                room_data = await response.json()
                created_room_name = room_data["name"]
                room_url = room_data["url"]

            logger.info(f"Created Daily room: {room_url}")

            # Generate tokens
            alice_token = await self._create_token(session, headers, created_room_name, "Alice")
            bob_token = await self._create_token(session, headers, created_room_name, "Bob")

            return RoomInfo(
                room_url=room_url,
                room_name=created_room_name,
                alice_token=alice_token,
                bob_token=bob_token
            )

    async def _create_token(
        self,
        session: aiohttp.ClientSession,
        headers: dict,
        room_name: str,
        participant_name: str
    ) -> str:
        """Create a meeting token for a participant."""
        token_config = {
            "properties": {
                "room_name": room_name,
                "user_name": participant_name,
                "is_owner": False,
                "enable_recording": False,
                "start_video_off": True,
                "start_audio_off": False,
            }
        }

        async with session.post(
            f"{self.DAILY_API_URL}/meeting-tokens",
            headers=headers,
            json=token_config
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to create token for {participant_name}: {response.status} - {error_text}")

            token_data = await response.json()
            logger.info(f"Generated token for {participant_name}")
            return token_data["token"]

    async def delete_room(self, room_name: str) -> bool:
        """Delete a Daily room."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.DAILY_API_URL}/rooms/{room_name}",
                headers=headers
            ) as response:
                if response.status in (200, 404):
                    logger.info(f"Deleted Daily room: {room_name}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to delete room {room_name}: {response.status} - {error_text}")
                    return False
