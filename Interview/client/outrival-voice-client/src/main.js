import "./style.css";
import { PipecatClient } from "@pipecat-ai/client-js";
import {
  WebSocketTransport,
  ProtobufFrameSerializer,
} from "@pipecat-ai/websocket-transport";

const app = document.querySelector("#app");

app.innerHTML = `
  <div style="font-family: system-ui; max-width: 760px; margin: 40px auto;">
    <h2>OutRival Interview Voice (Local)</h2>

    <p style="line-height:1.5">
      Paste the <code>ws_url</code> returned by your FastAPI <code>/connect</code> call,
      then click <b>Connect</b>. Your browser will ask for microphone permission.
    </p>

    <label for="wsUrl"><b>ws_url</b></label>
    <input id="wsUrl" style="width:100%; padding:10px; font-size:14px; margin-top:6px;"
      placeholder="ws://localhost:8000/ws/1/1/<conversation_uuid>"
    />

    <div style="margin-top:12px; display:flex; gap:10px;">
      <button id="btnConnect">Connect</button>
      <button id="btnDisconnect" disabled>Disconnect</button>
    </div>

    <div style="margin-top:14px; display:flex; gap:14px; align-items:center;">
      <span><b>Status:</b> <span id="status">idle</span></span>
    </div>

    <pre id="log" style="margin-top:14px; background:#111; color:#eee; padding:12px; border-radius:8px; overflow:auto; height: 280px;"></pre>
  </div>
`;

const wsUrlEl = document.querySelector("#wsUrl");
const btnConnect = document.querySelector("#btnConnect");
const btnDisconnect = document.querySelector("#btnDisconnect");
const statusEl = document.querySelector("#status");
const logEl = document.querySelector("#log");

function setStatus(s) {
  statusEl.textContent = s;
}
function log(...args) {
  logEl.textContent += args.join(" ") + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

let client = null;

btnConnect.onclick = async () => {
  const wsUrl = wsUrlEl.value.trim();
  if (!wsUrl) {
    log("Paste ws_url first (from /connect).");
    return;
  }

  setStatus("connecting...");
  btnConnect.disabled = true;

  client = new PipecatClient({
    transport: new WebSocketTransport({
      serializer: new ProtobufFrameSerializer(),

      // Start with 16000. If you get immediate disconnects or garbled audio,
      // we’ll switch both of these to 8000.
      recorderSampleRate: 16000,
      playerSampleRate: 16000,
    }),
    enableCam: false,
    enableMic: true,
    callbacks: {
      onBotConnected: () => {
        setStatus("connected");
        log("[bot] connected");
      },
      onBotReady: () => {
        log("[bot] ready (start talking)");
      },
      onBotDisconnected: () => {
        setStatus("disconnected");
        log("[bot] disconnected");
      },

      // Your side (already working)
      onUserTranscript: (t) => {
        const text = t?.text ?? t?.transcript ?? "";
        if (text) log("[you]", text);
      },

      // Ryan side (best-effort across versions)
      onBotTranscript: (t) => {
        const text = t?.text ?? t?.transcript ?? "";
        if (text) log("[ryan]", text);
      },

      // Some versions emit partials separately
      onBotPartialTranscript: (t) => {
        const text = t?.text ?? t?.transcript ?? "";
        if (text) log("[ryan…partial]", text);
      },

      // This is the safety net: show whatever the server is sending
      onServerMessage: (m) => {
        try {
          // Common patterns: message contains transcript text or event-type
          const type = m?.type || m?.event || m?.name || "server";
          const text =
            m?.text ||
            m?.transcript?.text ||
            m?.data?.text ||
            m?.data?.transcript?.text;

          if (text) {
            // If it smells like bot output, label it as Ryan
            const who =
              (m?.speaker && String(m.speaker).toLowerCase()) ||
              (m?.role && String(m.role).toLowerCase()) ||
              "";

            if (who.includes("assistant") || who.includes("bot") || who.includes("ryan")) {
              log("[ryan]", text);
            } else if (who.includes("user")) {
              log("[you]", text);
            } else {
              log(`[${type}]`, text);
            }
          } else {
            // Fallback: dump compact JSON for debugging
            log("[server]", JSON.stringify(m));
          }
        } catch (e) {
          log("[server]", String(m));
        }
      },

      onError: (e) => log("[error]", e?.message ?? String(e)),
    },

  });

  try {
    await client.connect({ wsUrl });
    btnDisconnect.disabled = false;
  } catch (err) {
    setStatus("failed");
    log("[connect failed]", err?.message ?? String(err));
    btnConnect.disabled = false;
    client = null;
  }
};

btnDisconnect.onclick = async () => {
  if (!client) return;
  try {
    await client.disconnect();
  } finally {
    btnDisconnect.disabled = true;
    btnConnect.disabled = false;
    setStatus("idle");
    client = null;
  }
};
