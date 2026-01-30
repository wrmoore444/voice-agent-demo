"""
=============================================================================
DAILY BOT-TO-BOT DEMO - Voice Agent Demo using Daily.co WebRTC
=============================================================================

A demonstration of two AI bots (Alice and Bob) having real-time voice
conversations through Daily.co WebRTC rooms.

NOTE: Requires Linux (or WSL on Windows) - daily-python is not available on Windows.

ENDPOINTS:
----------
GET  /daily-demo/personas     - List available persona files
POST /daily-demo/start        - Start a new conversation
POST /daily-demo/stop         - Stop the current conversation
GET  /daily-demo/status       - Get conversation state
WS   /daily-demo/viewer/ws    - WebSocket for real-time transcript updates
GET  /daily-demo/viewer       - HTML viewer page

REQUIRED ENVIRONMENT VARIABLES:
-------------------------------
- DAILY_API_KEY: Daily.co API key for room management
- GEMINI_API_KEY: For GeminiLiveLLMService
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager
from loguru import logger

from bot_demo_daily import DailyBotService, list_personas as daily_list_personas


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(title="Daily Bot-to-Bot Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DAILY BOT-TO-BOT DEMO ENDPOINTS
# =============================================================================

# Global service instance - handles one conversation at a time
daily_demo_service = DailyBotService()


@app.get("/daily-demo/personas")
async def get_daily_personas():
    """
    Get list of available personas for the Daily demo.

    Returns JSON with alice and bob persona lists:
    {
        "alice": ["alice_bank_teller", "alice_insurance_agent", ...],
        "bob": ["bob_bank_upset_customer", "bob_insurance_frustrated_claimant", ...]
    }
    """
    return daily_list_personas()


@app.post("/daily-demo/start")
async def start_daily_demo(
    topic: str = Query(default="", description="Conversation topic (optional)"),
    alice: str = Query(default=None, description="Alice persona filename (e.g., 'alice_bank_teller.json')"),
    bob: str = Query(default=None, description="Bob persona filename (e.g., 'bob_bank_upset_customer.json')"),
):
    """
    Start a Daily WebRTC bot-to-bot conversation.

    This creates a Daily room, has both bots join via WebRTC, and starts the conversation.
    Alice speaks first, then natural turn-taking occurs through Daily's audio mixing.

    Query Parameters:
        topic: Optional topic to seed the conversation
        alice: Alice persona filename (defaults to alice_insurance_agent.json)
        bob: Bob persona filename (defaults to bob_insurance_frustrated_claimant.json)

    Returns:
        JSON with status, selected personas, and room URL

    Errors:
        400: If a conversation is already running or failed to start

    Example:
        POST /daily-demo/start?alice=alice_bank_teller.json&bob=bob_bank_upset_customer.json
    """
    success = await daily_demo_service.start(
        topic=topic,
        alice_persona=alice,
        bob_persona=bob,
    )
    if not success:
        raise HTTPException(400, "Daily demo already running or failed to start")

    state = daily_demo_service.get_state()
    return {
        "status": "started",
        "implementation": "daily",
        "alice_persona": state.get("alice_persona"),
        "bob_persona": state.get("bob_persona"),
        "topic": state.get("topic"),
        "room_url": state.get("room_url"),
        "message": "Daily WebRTC bot conversation started"
    }


@app.post("/daily-demo/stop")
async def stop_daily_demo():
    """
    Stop the current Daily bot-to-bot conversation.

    This shuts down both bot pipelines and deletes the Daily room.

    Returns:
        JSON with status: "stopped"

    Errors:
        400: If no conversation is currently running
    """
    success = await daily_demo_service.stop()
    if not success:
        raise HTTPException(400, "No Daily demo running")
    return {
        "status": "stopped",
        "message": "Daily bot conversation stopped"
    }


@app.get("/daily-demo/status")
async def get_daily_demo_status():
    """
    Get the current state of the Daily demo.

    Returns JSON with:
        is_running: bool - Whether a conversation is active
        conversation_history: list - All messages in the conversation
        alice_persona: str - Current Alice persona filename
        bob_persona: str - Current Bob persona filename
        topic: str - Conversation topic (if set)
        room_url: str - Daily.co room URL (if running)
    """
    state = daily_demo_service.get_state()
    state["implementation"] = "daily"
    return state


@app.websocket("/daily-demo/viewer/ws")
async def daily_demo_viewer_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for viewing the Daily bot-to-bot conversation in real-time.

    MESSAGE TYPES SENT TO CLIENT:
    -----------------------------
    - {"type": "history", "data": [...]} - Full conversation history on connect
    - {"type": "message", "data": {...}} - New text message from Alice or Bob
    - {"type": "ping"}                   - Keepalive (every 30 seconds of inactivity)
    """
    await websocket.accept()
    viewer_queue = daily_demo_service.register_viewer()

    try:
        # Send current conversation history on connect
        state = daily_demo_service.get_state()
        await websocket.send_json({
            "type": "history",
            "data": state.get("conversation_history", [])
        })

        # Stream new messages
        while True:
            try:
                message = await asyncio.wait_for(viewer_queue.get(), timeout=30.0)
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})
            except Exception as e:
                logger.error(f"Error sending to Daily viewer: {e}")
                break
    except Exception as e:
        logger.error(f"Daily viewer WebSocket error: {e}")
    finally:
        daily_demo_service.unregister_viewer(viewer_queue)
        try:
            await websocket.close()
        except:
            pass


@app.get("/daily-demo/viewer")
async def daily_demo_viewer_page():
    """Serve the Daily demo viewer HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Bot-to-Bot Conversation</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #e2e8f0;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 8px;
            color: #38bdf8;
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 24px;
            font-size: 14px;
        }
        .persona-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .persona-box {
            background: #1e293b;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #334155;
        }
        .persona-box label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .persona-box.alice label {
            color: #a78bfa;
        }
        .persona-box.bob label {
            color: #2dd4bf;
        }
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #334155;
            border-radius: 8px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 14px;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #38bdf8;
        }
        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px 16px;
            border: 2px solid #334155;
            border-radius: 8px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 14px;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #38bdf8;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-start {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: white;
        }
        .btn-start:hover {
            opacity: 0.9;
        }
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        .btn-stop:hover {
            background: #dc2626;
        }
        .btn-clear {
            background: #334155;
            color: #e2e8f0;
        }
        .btn-clear:hover {
            background: #475569;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            background: #1e293b;
            border: 1px solid #334155;
        }
        .status.connected {
            border-color: #10b981;
            color: #10b981;
        }
        .status.disconnected {
            border-color: #ef4444;
            color: #ef4444;
        }
        .room-info {
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            background: #1e293b;
            border: 1px solid #38bdf8;
            display: none;
        }
        .room-info.visible {
            display: block;
        }
        .room-info a {
            color: #38bdf8;
            text-decoration: none;
        }
        .room-info a:hover {
            text-decoration: underline;
        }
        .conversation {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #334155;
        }
        .message {
            margin-bottom: 16px;
            padding: 14px 18px;
            border-radius: 12px;
            max-width: 80%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.alice {
            background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
            margin-right: auto;
        }
        .message.bob {
            background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
            margin-left: auto;
        }
        .message .speaker {
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
            opacity: 0.9;
        }
        .message .text {
            line-height: 1.5;
            font-size: 15px;
        }
        .message .time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 6px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
        }
        .empty-state p {
            margin-bottom: 10px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background: #38bdf8;
            color: #0f172a;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Daily Bot-to-Bot<span class="badge">Daily</span></h1>
        <p class="subtitle">Real-time voice conversation through Daily.co WebRTC</p>

        <div class="persona-row">
            <div class="persona-box alice">
                <label>Agent (Alice)</label>
                <select id="alicePersona">
                    <option>Loading personas...</option>
                </select>
            </div>
            <div class="persona-box bob">
                <label>Customer (Bob)</label>
                <select id="bobPersona">
                    <option>Loading personas...</option>
                </select>
            </div>
        </div>

        <div class="controls">
            <input type="text" id="topic" placeholder="Conversation topic (optional)...">
            <button class="btn-start" onclick="startDemo()">Start</button>
            <button class="btn-stop" onclick="stopDemo()">Stop</button>
            <button class="btn-clear" onclick="clearMessages()">Clear</button>
        </div>

        <div id="status" class="status disconnected">
            Disconnected
        </div>

        <div id="roomInfo" class="room-info">
            Join room (muted): <a id="roomUrl" href="#" target="_blank"></a>
        </div>

        <div id="conversation" class="conversation">
            <div class="empty-state">
                <p>No conversation yet.</p>
                <p>Select personas and click Start to begin.</p>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        const conversationDiv = document.getElementById('conversation');
        const statusDiv = document.getElementById('status');
        const roomInfoDiv = document.getElementById('roomInfo');

        function formatPersonaLabel(filename) {
            let name = filename.replace('.json', '').replace(/^(alice_|bob_)/, '');
            return name.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }

        async function loadPersonas() {
            try {
                const response = await fetch('/daily-demo/personas');
                const personas = await response.json();

                const aliceSelect = document.getElementById('alicePersona');
                const bobSelect = document.getElementById('bobPersona');

                aliceSelect.innerHTML = '';
                personas.alice.forEach((filename, index) => {
                    const option = document.createElement('option');
                    option.value = filename + '.json';
                    option.textContent = formatPersonaLabel(filename);
                    if (index === 0) option.selected = true;
                    aliceSelect.appendChild(option);
                });

                bobSelect.innerHTML = '';
                personas.bob.forEach((filename, index) => {
                    const option = document.createElement('option');
                    option.value = filename + '.json';
                    option.textContent = formatPersonaLabel(filename);
                    if (index === 0) option.selected = true;
                    bobSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading personas:', error);
            }
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/daily-demo/viewer/ws`);

            ws.onopen = () => {
                statusDiv.innerHTML = 'Connected - Watching Daily conversation';
                statusDiv.className = 'status connected';
            };

            ws.onclose = () => {
                statusDiv.innerHTML = 'Disconnected - Reconnecting...';
                statusDiv.className = 'status disconnected';
                setTimeout(connect, 2000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'history') {
                    if (data.data && data.data.length > 0) {
                        conversationDiv.innerHTML = '';
                        data.data.forEach(msg => addMessage(msg));
                    }
                } else if (data.type === 'message') {
                    addMessage(data.data);
                } else if (data.type === 'ping') {
                    // Keepalive
                }
            };
        }

        function addMessage(msg) {
            const emptyState = conversationDiv.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }

            const messageDiv = document.createElement('div');
            const speakerLower = msg.speaker.toLowerCase();
            const speakerClass = speakerLower === 'alice' ? 'alice' : 'bob';
            messageDiv.className = `message ${speakerClass}`;

            const time = new Date(msg.timestamp).toLocaleTimeString();

            messageDiv.innerHTML = `
                <div class="speaker">${msg.speaker}</div>
                <div class="text">${msg.text}</div>
                <div class="time">${time}</div>
            `;

            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        async function startDemo() {
            const topic = document.getElementById('topic').value;
            const alice = document.getElementById('alicePersona').value;
            const bob = document.getElementById('bobPersona').value;

            const params = new URLSearchParams();
            if (topic) params.append('topic', topic);
            if (alice) params.append('alice', alice);
            if (bob) params.append('bob', bob);

            try {
                const response = await fetch(`/daily-demo/start?${params.toString()}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    const aliceLabel = formatPersonaLabel(alice);
                    const bobLabel = formatPersonaLabel(bob);
                    statusDiv.innerHTML = `Started - ${aliceLabel} vs ${bobLabel}`;

                    if (data.room_url) {
                        const roomUrlLink = document.getElementById('roomUrl');
                        roomUrlLink.href = data.room_url;
                        roomUrlLink.textContent = data.room_url;
                        roomInfoDiv.classList.add('visible');
                    }
                } else {
                    alert(data.detail || 'Failed to start demo');
                }
            } catch (error) {
                console.error('Error starting demo:', error);
                alert('Failed to start demo');
            }
        }

        async function stopDemo() {
            try {
                const response = await fetch('/daily-demo/stop', { method: 'POST' });
                const data = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = 'Stopped';
                    roomInfoDiv.classList.remove('visible');
                } else {
                    alert(data.detail || 'Failed to stop demo');
                }
            } catch (error) {
                console.error('Error stopping demo:', error);
                alert('Failed to stop demo');
            }
        }

        function clearMessages() {
            conversationDiv.innerHTML = `
                <div class="empty-state">
                    <p>No conversation yet.</p>
                    <p>Select personas and click Start to begin.</p>
                </div>
            `;
            roomInfoDiv.classList.remove('visible');
        }

        // Initialize on page load
        loadPersonas();
        connect();
    </script>
</body>
</html>
    """
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
