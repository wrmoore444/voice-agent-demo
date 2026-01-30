# Voice Agent Demo

A demonstration of AI-powered voice agents using Pipecat, featuring bot-to-bot conversations with realistic personas.

## Features

- **Bot-to-Bot Demo (Pipecat)**: Two AI agents converse using ElevenLabs text-to-speech
- **Bot-to-Bot Demo (Daily)**: Two AI agents converse via Daily.co WebRTC with Gemini Live native voice
- **Human-to-Agent Demo**: Talk to an AI agent using your microphone
- **Persona System**: Pre-built scenarios for banking, insurance, and travel industries
- **Web Interface**: Interactive viewer with real-time conversation display

## Requirements

### WSL Required (Windows Users)

**This application must be run from WSL (Windows Subsystem for Linux)**, not native Windows. The Daily.co WebRTC library (`daily-python`) is not available on Windows.

If you don't have WSL installed:
```bash
wsl --install
```

### Python Version

**Python 3.10, 3.11, 3.12, or 3.13 required** (Python 3.14+ is NOT supported due to Pipecat compatibility)

Check your version:
```bash
python3 --version
```

### API Keys Required

| API Key | Required For | Get It From |
|---------|-------------|-------------|
| `GEMINI_API_KEY` | All demos | [Google AI Studio](https://aistudio.google.com/) |
| `DAILY_API_KEY` | Daily bot-to-bot demo | [Daily.co Dashboard](https://dashboard.daily.co/) |
| `ELEVENLABS_API_KEY` | Pipecat audio mode | [ElevenLabs](https://elevenlabs.io/) |

## Quick Start (WSL/Linux)

### 1. Navigate to Project

```bash
cd voice-agent-demo/Interview
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you see errors about `daily-python`, ensure you're in WSL, not Windows.

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required for all demos
GEMINI_API_KEY=your_gemini_api_key_here

# Required for Daily bot-to-bot demo
DAILY_API_KEY=your_daily_api_key_here

# Required for Pipecat audio mode
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Database (SQLite works out of the box)
DATABASE_URL=sqlite+aiosqlite:///./interview.db
```

### 5. Run the Server

```bash
python main.py
```

### 6. Open the Demo

Open your browser to one of these URLs:

| Demo | URL |
|------|-----|
| **Pipecat Bot-to-Bot** | http://localhost:8001/pipecat-demo/viewer |
| **Daily Bot-to-Bot** | http://localhost:8001/daily-demo/viewer |
| **Talk to Alice** | http://localhost:8001/human-demo/viewer |

## Demo Descriptions

### Pipecat Bot-to-Bot (`/pipecat-demo/viewer`)

Two AI agents (Alice and Bob) have a text-based conversation. Optionally enable audio to hear them speak using ElevenLabs voices.

**Best for:** Testing persona behavior, no microphone needed

### Daily Bot-to-Bot (`/daily-demo/viewer`)

Two AI agents converse using real-time WebRTC audio via Daily.co. Uses Gemini Live's native voice capabilities.

**Best for:** Demonstrating real voice AI, testing latency

### Talk to Alice (`/human-demo/viewer`)

Have a live voice conversation with Alice (an AI customer service agent) using your microphone.

**Best for:** Interactive demo, testing human-to-AI interaction

## Available Personas

### Alice (Agent)
- `alice_bank_teller` - Friendly bank representative
- `alice_insurance_agent` - Professional insurance agent
- `alice_travel_agent` - Helpful travel consultant

### Bob (Customer)
- `bob_bank_new_customer` - First-time bank customer
- `bob_bank_upset_customer` - Frustrated about fees
- `bob_insurance_new_customer` - Shopping for coverage
- `bob_insurance_frustrated_claimant` - Unhappy with claim process
- `bob_travel_first_time` - Planning first trip
- `bob_wrong_number` - Quick test scenario (wrong number call)

## Troubleshooting

### "Address already in use" error
Kill the old process and restart:
```bash
pkill -f python
python main.py
```

### "daily-python not found" or similar
You're running on Windows. Switch to WSL:
```bash
wsl
cd /path/to/project/Interview
```

### "No audio playing" (Pipecat demo)
- Verify `ELEVENLABS_API_KEY` is set in `.env`
- Check the "Enable Audio" checkbox in the viewer

### "Conversation not starting"
- Verify `GEMINI_API_KEY` is set in `.env`
- Check the terminal for error messages

### "Daily demo not working"
- Verify `DAILY_API_KEY` is set in `.env`
- Ensure you're running from WSL, not Windows

## Project Structure

```
Interview/
├── main.py                 # FastAPI server (entry point)
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── models.py              # Database models
├── voice_agent.py         # Human-to-agent voice handler
├── bot_demo_daily/        # Daily WebRTC bot-to-bot demo
│   ├── daily_bot_service.py
│   ├── bot_pipeline_factory.py
│   └── daily_room_manager.py
├── bot_demo_pipecat/      # Pipecat bot-to-bot demo
│   ├── dual_bot_service.py
│   ├── persona_loader.py
│   └── processors/
└── bot_demo/
    └── personas/          # Persona JSON files
```

## Making Code Changes

Use [Claude Code](https://claude.ai/claude-code) for AI-assisted code modifications:

```bash
cd voice-agent-demo
claude
```

Or use [claude.ai](https://claude.ai) in your browser to describe changes and copy/paste code.
