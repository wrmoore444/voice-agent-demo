# Pipecat Bot-to-Bot Demo API Reference

This document describes the REST API and WebSocket endpoints for the Pipecat bot-to-bot conversation demo.

## Base URL

```
http://localhost:8000
```

## Quick Start

1. Start the server: `python main.py` (from the Interview folder)
2. Open browser: `http://localhost:8000/pipecat-demo/viewer`
3. Select personas, click "Start"

---

## REST Endpoints

### GET /pipecat-demo/personas

List available persona files for Alice and Bob.

**Response:**
```json
{
  "alice": ["alice_bank_teller", "alice_insurance_agent", "alice_travel_agent"],
  "bob": ["bob_bank_upset_customer", "bob_insurance_frustrated_claimant", ...]
}
```

**Example:**
```bash
curl http://localhost:8000/pipecat-demo/personas
```

---

### POST /pipecat-demo/start

Start a new bot-to-bot conversation.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | string | "" | Optional topic to seed the conversation |
| `alice` | string | alice_insurance_agent.json | Alice persona filename |
| `bob` | string | bob_insurance_frustrated_claimant.json | Bob persona filename |
| `enable_audio` | boolean | false | Enable ElevenLabs TTS audio generation |

**Response:**
```json
{
  "status": "started",
  "implementation": "pipecat",
  "alice_persona": "alice_bank_teller.json",
  "bob_persona": "bob_bank_upset_customer.json",
  "topic": "",
  "audio_enabled": true,
  "message": "Pipecat bot conversation started"
}
```

**Errors:**
- `400`: Conversation already running or failed to start

**Examples:**
```bash
# Start with defaults (no audio)
curl -X POST "http://localhost:8000/pipecat-demo/start"

# Start with specific personas and audio
curl -X POST "http://localhost:8000/pipecat-demo/start?alice=alice_bank_teller.json&bob=bob_bank_upset_customer.json&enable_audio=true"

# Start with a topic
curl -X POST "http://localhost:8000/pipecat-demo/start?topic=overdraft%20fees&enable_audio=true"
```

---

### POST /pipecat-demo/stop

Stop the current conversation gracefully.

**Response:**
```json
{
  "status": "stopped",
  "message": "Pipecat bot conversation stopped"
}
```

**Errors:**
- `400`: No conversation is currently running

**Note:** Stop triggers a graceful shutdown. TTS workers will finish processing any queued audio before stopping completely.

**Example:**
```bash
curl -X POST "http://localhost:8000/pipecat-demo/stop"
```

---

### GET /pipecat-demo/status

Get the current state of the demo.

**Response:**
```json
{
  "is_running": true,
  "turn_count": 5,
  "conversation_history": [
    {
      "speaker": "Alice",
      "text": "Good morning! Welcome to First National Bank...",
      "timestamp": "2026-01-27T20:30:00.000Z",
      "pace": 0.5,
      "energy": "normal"
    },
    ...
  ],
  "audio_enabled": true,
  "tts_queue_size": 2,
  "alice_persona": "alice_bank_teller.json",
  "bob_persona": "bob_bank_upset_customer.json",
  "topic": "",
  "implementation": "pipecat"
}
```

**Example:**
```bash
curl http://localhost:8000/pipecat-demo/status
```

---

## WebSocket Endpoint

### WS /pipecat-demo/viewer/ws

Real-time streaming of conversation messages and audio.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/pipecat-demo/viewer/ws');
```

**Message Types Received:**

#### 1. History (on connect)
Sent immediately after connection with full conversation history.
```json
{
  "type": "history",
  "data": [
    {"speaker": "Alice", "text": "...", "timestamp": "...", ...},
    {"speaker": "Bob", "text": "...", "timestamp": "...", ...}
  ]
}
```

#### 2. Message (new text)
Sent when a bot speaks.
```json
{
  "type": "message",
  "data": {
    "speaker": "Alice",
    "text": "Good morning! Welcome to First National Bank.",
    "timestamp": "2026-01-27T20:30:00.000Z",
    "pace": 0.5,
    "energy": "normal",
    "overlap_ms": 200
  }
}
```

#### 3. Audio (TTS output)
Sent when audio is ready for playback.
```json
{
  "type": "audio",
  "data": {
    "speaker": "Alice",
    "audio": "<base64-encoded PCM data>",
    "format": "pcm",
    "sample_rate": 24000,
    "sequence": 1,
    "pace": 0.5,
    "energy": "normal",
    "overlap_ms": 200,
    "timestamp": "2026-01-27T20:30:01.000Z"
  }
}
```

**Audio Data Details:**
- `audio`: Base64-encoded raw PCM audio
- `format`: Always "pcm"
- `sample_rate`: 24000 Hz (ElevenLabs default)
- `sequence`: Global ordering number for playback sequencing

#### 4. Ping (keepalive)
Sent every 30 seconds of inactivity.
```json
{
  "type": "ping"
}
```

---

## HTML Viewer Page

### GET /pipecat-demo/viewer

Serves the interactive viewer page. This is an embedded HTML page (not a separate file) that provides:

**Features:**
- Persona selection dropdowns (populated from /pipecat-demo/personas)
- Audio toggle checkbox
- Start / Stop / Clear buttons
- Real-time conversation display with speaker avatars
- Audio playback with visual "speaking" indicators
- Connection status display

**How It Works:**

1. **On Load:**
   - Fetches personas from `/pipecat-demo/personas`
   - Populates Alice and Bob dropdown selectors
   - Connects to WebSocket `/pipecat-demo/viewer/ws`

2. **On Start Click:**
   - Reads selected personas and audio toggle
   - POSTs to `/pipecat-demo/start` with query parameters
   - WebSocket begins receiving messages

3. **Message Handling:**
   - `history`: Renders all past messages
   - `message`: Appends new message to conversation
   - `audio`: Queues audio for playback

4. **Audio Playback:**
   - Uses Web Audio API (`AudioContext`)
   - Decodes base64 PCM to audio buffer
   - Plays in sequence order (handles out-of-order arrival)
   - Shows "speaking" indicator during playback

5. **On Stop Click:**
   - POSTs to `/pipecat-demo/stop`
   - Conversation stops, audio queue drains

**Styling:**
- Dark theme with gradient background
- Pink accent for Alice, Cyan accent for Bob
- Responsive layout
- Animated speaking indicators

---

## Environment Variables

Required for audio generation:

```bash
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID_ALICE=21m00Tcm4TlvDq8ikWAM  # Optional, default: Rachel
ELEVENLABS_VOICE_ID_BOB=ErXwobaYiN019PkySvjV    # Optional, default: Antoni
```

Required for LLM:

```bash
GEMINI_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

---

## Example: Programmatic Control

Start a conversation and poll for status:

```python
import requests
import time

BASE = "http://localhost:8000"

# Start conversation
response = requests.post(f"{BASE}/pipecat-demo/start", params={
    "alice": "alice_bank_teller.json",
    "bob": "bob_bank_upset_customer.json",
    "enable_audio": False
})
print(response.json())

# Poll status
while True:
    status = requests.get(f"{BASE}/pipecat-demo/status").json()
    print(f"Turn {status['turn_count']}, Running: {status['is_running']}")

    if not status['is_running']:
        break
    time.sleep(2)

# Get final conversation
final = requests.get(f"{BASE}/pipecat-demo/status").json()
for msg in final['conversation_history']:
    print(f"{msg['speaker']}: {msg['text']}")
```

---

## Example: WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8000/pipecat-demo/viewer/ws');

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
        case 'history':
            console.log('Received history:', msg.data.length, 'messages');
            break;
        case 'message':
            console.log(`${msg.data.speaker}: ${msg.data.text}`);
            break;
        case 'audio':
            console.log(`Audio for ${msg.data.speaker}: ${msg.data.audio.length} bytes`);
            // Decode and play audio...
            break;
        case 'ping':
            // Keepalive, ignore
            break;
    }
};

// Start conversation after connecting
ws.onopen = () => {
    fetch('/pipecat-demo/start?enable_audio=true', { method: 'POST' });
};
```

---

## Error Handling

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad request (already running, not running, invalid params) |
| 500 | Server error (check logs) |

Common issues:
- **"Already running"**: Call `/stop` first, or wait for conversation to end
- **No audio**: Check `ELEVENLABS_API_KEY` is set
- **WebSocket disconnects**: Network issue or server restart; client should reconnect
