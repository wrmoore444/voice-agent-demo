"""Pace analyzer processor for analyzing conversation energy."""

from typing import Optional
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)

from ..pace_analyzer import analyze_pace, pace_to_overlap_ms, PaceAnalysis


class PaceAnalyzerProcessor(FrameProcessor):
    """
    Custom FrameProcessor that analyzes the pace/energy of LLM responses.

    This processor:
    - Accumulates text during LLM responses
    - Analyzes pace when response completes
    - Can notify a bridge processor with pace information
    """

    def __init__(
        self,
        bot_name: str,
        bridge_processor: Optional["BotBridgeProcessor"] = None,
        **kwargs
    ):
        """
        Initialize the pace analyzer processor.

        Args:
            bot_name: Name of this bot for logging
            bridge_processor: Optional bridge processor to notify with pace info
        """
        super().__init__(**kwargs)
        self.bot_name = bot_name
        self.bridge_processor = bridge_processor

        # Accumulate text during LLM response
        self._accumulating = False
        self._accumulated_text = ""
        self._last_analysis: Optional[PaceAnalysis] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and analyze pace."""
        await super().process_frame(frame, direction)

        # Start accumulating when LLM response begins
        if isinstance(frame, LLMFullResponseStartFrame):
            self._accumulating = True
            self._accumulated_text = ""

        # Accumulate text frames
        elif isinstance(frame, TextFrame) and self._accumulating:
            self._accumulated_text += frame.text

        # Analyze pace when response ends
        elif isinstance(frame, LLMFullResponseEndFrame) and self._accumulating:
            self._accumulating = False

            if self._accumulated_text.strip():
                self._analyze_and_notify(self._accumulated_text.strip())

            self._accumulated_text = ""

        # Always pass frames downstream
        await self.push_frame(frame, direction)

    def _analyze_and_notify(self, text: str):
        """Analyze text pace and notify bridge processor."""
        analysis = analyze_pace(text)
        overlap_ms = pace_to_overlap_ms(analysis.pace)

        self._last_analysis = analysis

        logger.debug(
            f"[{self.bot_name}] Pace: {analysis.pace:.2f} ({analysis.energy}) "
            f"overlap: {overlap_ms}ms - {analysis.reason}"
        )

        # Notify bridge processor if available
        if self.bridge_processor:
            self.bridge_processor.set_pace_info(
                pace=analysis.pace,
                energy=analysis.energy,
                overlap_ms=overlap_ms
            )

    @property
    def last_analysis(self) -> Optional[PaceAnalysis]:
        """Get the last pace analysis."""
        return self._last_analysis
