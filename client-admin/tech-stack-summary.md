# Technology Stack Summary

Prepared for client discussion.

---

## 1. FastAPI

### Why FastAPI for this app?

**Native async support** - Voice agents require real-time audio streaming, WebSocket connections, and concurrent API calls. FastAPI is async-first, handling many simultaneous connections without blocking.

**Built-in WebSocket support** - Real-time voice communication needs WebSockets. FastAPI handles this cleanly out of the box.

**Pipecat compatibility** - Pipecat is designed around Python's async/await patterns. FastAPI speaks the same language.

**Performance** - One of the fastest Python frameworks, close to Node.js and Go in benchmarks. For real-time audio, latency matters.

**Automatic API documentation** - Swagger UI comes free for API exploration and frontend integration.

### Why not Flask?

Flask is synchronous by default. For real-time voice streaming with Pipecat, you'd be fighting uphill - bolting on async support, adding WebSocket libraries, and wrapping async code awkwardly. Flask is fine for traditional request/response APIs, but FastAPI is the natural fit here.

### Common questions

- **"Is it mature?"** - Yes. Used by Microsoft, Netflix, Uber. Production-ready since 2018.
- **"What about Django?"** - Django is great for content-heavy sites with admin panels. For API-first apps with real-time requirements, FastAPI is leaner and faster.
- **"How does it scale?"** - Horizontally like any stateless API. Multiple uvicorn workers or container replicas.

### Things to know

- Deployment uses ASGI servers (uvicorn), not traditional WSGI
- Async can be "contagious" - database calls, HTTP clients should also be async
- No batteries included (no admin panel, no built-in auth) - you build what you need

---

## 2. Pipecat

### What is Pipecat?

An open-source framework for building real-time voice and multimodal AI agents. It provides the plumbing for:

```
Audio in -> Speech-to-Text -> LLM -> Text-to-Speech -> Audio out
```

Plus the hard parts: interruption handling, voice activity detection, turn-taking, buffering.

### Why Pipecat?

**Purpose-built** - Designed from the ground up for conversational AI agents, not adapted from something else.

**Pipeline architecture** - Clean mental model. Compose processors in a chain. Easy to swap components.

**Handles the hard stuff** - Interruption detection, voice activity detection (VAD), audio buffering, turn-taking. Building this from scratch is months of work.

**Broad integrations**:
- LLMs: OpenAI, Anthropic, Google Gemini
- TTS: ElevenLabs, Cartesia, Google, Azure, Deepgram
- STT: Deepgram, Google, Whisper
- Transport: WebSockets, WebRTC, Daily

**Open source (Apache 2.0)** - No vendor lock-in. Backed by Daily.co but not dependent on their platform.

### Alternatives

| Option | Type | Trade-off |
|--------|------|-----------|
| Build from scratch | DIY | Full control, but months of work on interruptions, buffering, timing |
| LiveKit Agents | Open source | Similar concept, more WebRTC/video focused |
| Vocode | Open source | Comparable, different API, less active |
| Retell AI | Managed | Faster start, less control, per-minute pricing |
| Vapi | Managed | Similar to Retell |
| Bland AI | Managed | Phone-call focused |

### The build vs. buy decision

**Open source (Pipecat)**: More control, you host it, lower marginal cost at scale.

**Managed platforms (Retell, Vapi)**: Faster to prototype, less control, per-minute fees add up, dependent on their uptime.

Pipecat hits the sweet spot: open source control without starting from zero.

### Things to know

- Still maturing - API changes between versions
- Debugging real-time audio pipelines is tricky
- Learning curve for pipeline concepts and async patterns
- Documentation improving but sometimes requires reading source

---

## 3. Implementation Overview

### Pipeline architecture

```
WebSocket Input -> RTVI -> Context (user) -> Transcript -> LLM -> [TTS] -> Transcript -> WebSocket Output -> Context (assistant)
```

### Key components

| Component | Purpose |
|-----------|---------|
| FastAPIWebsocketTransport | Bridges FastAPI WebSockets to Pipecat |
| SileroVADAnalyzer | Voice Activity Detection - knows when user is speaking |
| GeminiLiveLLMService | Gemini with native audio (no separate STT needed) |
| ElevenLabsTTSService | Text-to-speech (conditional, per-agent) |
| TranscriptProcessor | Captures conversation for storage/analysis |
| RTVIProcessor | Real-Time Voice Interface protocol |
| MinWordsInterruptionStrategy | User must say 2+ words to interrupt |

### Design highlights

**Dual pipeline modes** - Some agents use Gemini's native audio output, others use ElevenLabs TTS. Shows component flexibility.

**Memory management** - Transcriptions save in batches of 10 to prevent memory buildup. Careful cleanup of ORM objects.

**Interruption handling** - MinWordsInterruptionStrategy(min_words=2) prevents false interruptions from background noise.

**LLM function calling** - The `end_conversation` function lets the LLM decide when to hang up gracefully.

**Post-call processing** - After calls end, datapoints are extracted and emailed for lead capture.

### Potential questions

- **"Why Gemini Live?"** - Native audio-to-audio reduces latency. Could swap to OpenAI if needed.
- **"What if Gemini goes down?"** - Pipecat's abstraction allows provider swapping.
- **"Why ElevenLabs for just one agent?"** - Voice quality preferences or specific requirements per agent.

---

*Summary prepared January 2026*
