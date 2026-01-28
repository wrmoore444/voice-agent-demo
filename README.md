# Voice Agent Demo

A demonstration of AI-powered voice agents using Pipecat, featuring bot-to-bot conversations with realistic personas.

## Features

- **Bot-to-Bot Demo**: Watch two AI agents (Alice and Bob) have natural conversations using customizable personas
- **Real-time Audio**: ElevenLabs text-to-speech with conversation pacing
- **Persona System**: Pre-built scenarios for banking, insurance, and travel industries
- **Web Interface**: Interactive viewer with real-time conversation display

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/wrmoore444/voice-agent-demo.git
cd voice-agent-demo/Interview
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required - Google Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Required for audio - ElevenLabs API key
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional - custom voices for Alice and Bob
ELEVENLABS_VOICE_ID_ALICE=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_VOICE_ID_BOB=ErXwobaYiN019PkySvjV
```

### 5. Run the Server

```bash
python main.py
```

### 6. Open the Demo

Open your browser to: **http://localhost:8000/pipecat-demo/viewer**

## Using the Demo

1. **Select Personas**: Choose an Alice persona (the agent) and a Bob persona (the customer)
2. **Enable Audio** (optional): Check the "Enable Audio" box for voice output
3. **Click Start**: The conversation begins automatically
4. **Watch**: Messages appear in real-time with optional audio playback
5. **Click Stop**: End the conversation at any time

## Available Personas

### Alice (Agent)
- `alice_bank_teller` - Friendly bank representative
- `alice_insurance_agent` - Professional insurance agent
- `alice_travel_agent` - Helpful travel consultant

### Bob (Customer)
- `bob_bank_new_customer` - First-time bank customer
- `bob_bank_upset_customer` - Frustrated about fees
- `bob_bank_confused_customer` - Needs help understanding services
- `bob_insurance_new_customer` - Shopping for coverage
- `bob_insurance_frustrated_claimant` - Unhappy with claim process
- `bob_travel_first_time` - Planning first trip
- `bob_travel_angry_customer` - Had a bad experience

## API Documentation

See [API_REFERENCE.md](Interview/bot_demo_pipecat/API_REFERENCE.md) for REST and WebSocket endpoints.

## Project Structure

```
voice-agent-demo/
├── Interview/
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment template
│   ├── bot_demo_pipecat/      # Pipecat bot-to-bot demo
│   │   ├── dual_bot_service.py
│   │   ├── persona_loader.py
│   │   ├── API_REFERENCE.md
│   │   └── processors/
│   └── bot_demo/              # Original bot demo
│       └── personas/          # Persona JSON files
└── client-admin/              # Admin tools (optional)
```

## Troubleshooting

### "No audio playing"
- Verify `ELEVENLABS_API_KEY` is set in `.env`
- Check the "Enable Audio" checkbox in the viewer
- Check browser console for errors

### "Conversation not starting"
- Verify `GEMINI_API_KEY` is set in `.env`
- Check the terminal for error messages

### "Cannot connect to server"
- Ensure `python main.py` is running
- Check that port 8000 is not in use

## Requirements

- Python 3.10+
- API Keys:
  - [Google AI Studio](https://aistudio.google.com/) - Gemini API key
  - [ElevenLabs](https://elevenlabs.io/) - For audio (optional)
