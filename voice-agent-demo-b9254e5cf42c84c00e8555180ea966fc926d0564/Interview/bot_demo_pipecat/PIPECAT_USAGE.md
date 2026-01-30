# Pipecat Functionality Used in bot_demo_pipecat

This document catalogs all Pipecat components, patterns, and functionality used in the `bot_demo_pipecat` module.

## Overview

The `bot_demo_pipecat` module implements a bot-to-bot conversation demo using Pipecat's architecture patterns. While the current implementation uses direct Gemini API calls for text generation (since `GeminiLiveLLMService` is optimized for streaming audio), the custom processors follow Pipecat's `FrameProcessor` pattern and are ready for full pipeline integration when TTS is re-enabled.

---

## Pipecat Imports by File

### `dual_bot_service.py`

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import TextFrame, LLMRunFrame, EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams,
    GeminiModalities,
)
```

### `processors/bridge_processor.py`

```python
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
```

### `processors/turn_processor.py`

```python
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    LLMFullResponseEndFrame,
)
```

### `processors/pace_processor.py`

```python
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
```

---

## Pipecat Components Explained

### Core Pipeline Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `Pipeline` | `pipecat.pipeline.pipeline` | Linear chain of processors that frames flow through |
| `PipelineRunner` | `pipecat.pipeline.runner` | Async executor for pipeline tasks |
| `PipelineTask` | `pipecat.pipeline.task` | Wraps a pipeline with execution parameters |
| `PipelineParams` | `pipecat.pipeline.task` | Configuration for task execution (interruptions, metrics) |

### Frame Types

| Frame | Module | Purpose |
|-------|--------|---------|
| `Frame` | `pipecat.frames.frames` | Base class for all frames |
| `TextFrame` | `pipecat.frames.frames` | Contains text content from LLM or other sources |
| `LLMRunFrame` | `pipecat.frames.frames` | Triggers LLM to generate a response |
| `LLMFullResponseStartFrame` | `pipecat.frames.frames` | Signals start of complete LLM response |
| `LLMFullResponseEndFrame` | `pipecat.frames.frames` | Signals end of complete LLM response |
| `EndTaskFrame` | `pipecat.frames.frames` | Signals pipeline task should terminate |

### Processors

| Component | Module | Purpose |
|-----------|--------|---------|
| `FrameProcessor` | `pipecat.processors.frame_processor` | Base class for custom processors |
| `FrameDirection` | `pipecat.processors.frame_processor` | Enum for UPSTREAM/DOWNSTREAM routing |
| `OpenAILLMContext` | `pipecat.processors.aggregators.openai_llm_context` | Manages conversation context/history |

### LLM Services

| Component | Module | Purpose |
|-----------|--------|---------|
| `GeminiLiveLLMService` | `pipecat.services.google.gemini_live.llm` | Google Gemini Live API integration |
| `InputParams` | `pipecat.services.google.gemini_live.llm` | Configuration for Gemini (temperature, etc.) |
| `GeminiModalities` | `pipecat.services.google.gemini_live.llm` | TEXT or AUDIO output mode selection |

---

## Custom Processors (Pipecat Pattern)

### BotBridgeProcessor

**Extends:** `FrameProcessor`

**Purpose:** Routes LLM output between bot pipelines and broadcasts to viewers.

**Frame Handling:**
- `LLMFullResponseStartFrame` → Start accumulating text
- `TextFrame` → Accumulate text content
- `LLMFullResponseEndFrame` → Route complete message to partner bot

**Key Methods:**
```python
async def process_frame(self, frame: Frame, direction: FrameDirection)
async def _route_message(self, text: str)
def set_pace_info(self, pace: float, energy: str, overlap_ms: int)
```

### TurnControlProcessor

**Extends:** `FrameProcessor`

**Purpose:** Enforces turn-taking by gating `LLMRunFrame` until it's the bot's turn.

**Frame Handling:**
- `LLMRunFrame` → Wait for turn, then pass through
- `LLMFullResponseEndFrame` → Signal turn complete

**Key Methods:**
```python
async def process_frame(self, frame: Frame, direction: FrameDirection)
async def _wait_for_our_turn(self) -> bool
```

**Shared State:** Uses `TurnState` dataclass for coordination between two processors.

### PaceAnalyzerProcessor

**Extends:** `FrameProcessor`

**Purpose:** Analyzes conversation energy/pace for natural audio timing.

**Frame Handling:**
- `LLMFullResponseStartFrame` → Start accumulating
- `TextFrame` → Accumulate text
- `LLMFullResponseEndFrame` → Analyze and notify bridge

**Key Methods:**
```python
async def process_frame(self, frame: Frame, direction: FrameDirection)
def _analyze_and_notify(self, text: str)
```

---

## Pipecat Patterns Used

### 1. Frame Processing Pattern

All custom processors follow the standard pattern:

```python
class CustomProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Handle specific frame types
        if isinstance(frame, SomeFrameType):
            # Process frame
            pass

        # Always pass frames downstream (unless filtering)
        await self.push_frame(frame, direction)
```

### 2. Response Accumulation Pattern

Used to collect streaming text into complete responses:

```python
# On LLMFullResponseStartFrame
self._accumulating = True
self._accumulated_text = ""

# On TextFrame (while accumulating)
self._accumulated_text += frame.text

# On LLMFullResponseEndFrame
self._accumulating = False
# Process complete text
```

### 3. Pipeline Construction Pattern

```python
pipeline = Pipeline([
    transport.input(),
    context_aggregator.user(),
    turn_processor,        # Custom: gate LLM triggers
    llm_service,           # GeminiLiveLLMService
    pace_processor,        # Custom: analyze energy
    bridge_processor,      # Custom: route messages
    transport.output(),
    context_aggregator.assistant(),
])

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
    ),
)

runner = PipelineRunner()
await runner.run(task)
```

---

## Current Implementation Status

### Active (Used Now)
- `FrameProcessor` base class pattern
- `Frame` type hierarchy for typing
- `FrameDirection` enum
- Processor architecture pattern

### Prepared (Ready for TTS Integration)
- `Pipeline` construction
- `PipelineRunner` execution
- `GeminiLiveLLMService` with `GeminiModalities.TEXT`
- `OpenAILLMContext` for conversation history

### Future (When TTS Re-enabled)
- Full pipeline execution with audio output
- `GeminiModalities.AUDIO` for voice responses
- Transport layer integration

---

## Related Pipecat Components (Not Yet Used)

These components from `voice_agent.py` could be integrated in future:

| Component | Purpose |
|-----------|---------|
| `FastAPIWebsocketTransport` | WebSocket I/O for pipelines |
| `SileroVADAnalyzer` | Voice activity detection |
| `ElevenLabsTTSService` | Text-to-speech synthesis |
| `TranscriptProcessor` | Conversation transcription |
| `RTVIProcessor` | Real-time voice interface |
| `MinWordsInterruptionStrategy` | Interruption handling |

---

## References

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- Existing usage: `voice_agent.py` in this project
