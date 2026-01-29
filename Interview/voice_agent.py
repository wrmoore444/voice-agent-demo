import sys
import asyncio
from dotenv import load_dotenv
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from email_utils import send_call_summary_email
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import TTSSpeakFrame, EndTaskFrame,LLMRunFrame,TranscriptionMessage
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties
import os
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.audio.interruptions.min_words_interruption_strategy import (
    MinWordsInterruptionStrategy,
)
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams,
    GeminiModalities,
)
from datapoint_extractor import extract_datapoints_from_conversation
import logging
import sys
from loguru import logger as loguru_logger

load_dotenv(override=True)
loguru_logger.remove()

LOG_FORMAT = "%(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, ERROR as needed
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),  # Logs to console
    ],
)
logger = logging.getLogger("inbound-ai")


end_conversation_fn = FunctionSchema(
    name="end_conversation",
    description="use this function to end the conversation when the user say goodbye, take the user consent into consideration before ending the call",
    properties={
    },
    required=[],
)

tools_schema = ToolsSchema(standard_tools=[
    end_conversation_fn
])


# Default prompt if agent doesn't have one configured
DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful AI voice assistant. 
Engage naturally in conversation, answer questions clearly, and be friendly and professional.

ENDING THE CALL:
You have access to an `end_conversation` function that disconnects the call. You MUST call this function to end the call properly.

When to call `end_conversation`:
- After the customer says goodbye, thank you, or indicates they're done (e.g., "bye", "thanks, that's all", "have a good day", "I'm all set")
- After YOU have said your brief farewell

How to end the call:
1. Say a brief, warm goodbye (e.g., "Have a great day!" or "Thank you for calling, goodbye!")
2. IMMEDIATELY call the `end_conversation` function - do NOT wait for another response

IMPORTANT: Once goodbyes have been exchanged, call `end_conversation` right away. Do not continue talking or ask additional questions after the customer has said goodbye.
"""


def create_voice_system_instruction(agent_prompt: str = None, agent_name: str = None):
    """Create system instruction from agent's custom prompt or use default"""
    end_call_instructions = """

ENDING THE CALL:
You have access to an `end_conversation` function that disconnects the call. You MUST call this function to end the call properly.

When to call `end_conversation`:
- After the customer says goodbye, thank you, or indicates they're done (e.g., "bye", "thanks, that's all", "have a good day", "I'm all set")
- After YOU have said your brief farewell

How to end the call:
1. Say a brief, warm goodbye (e.g., "Have a great day!" or "Thank you for calling, goodbye!")
2. IMMEDIATELY call the `end_conversation` function - do NOT wait for another response

IMPORTANT: Once goodbyes have been exchanged, call `end_conversation` right away. Do not continue talking or ask additional questions after the customer has said goodbye.
"""
    if agent_prompt:
        # Use the agent's custom prompt
        instruction = agent_prompt
        if agent_name:
            instruction = f"You are {agent_name}.\n\n{instruction}"
        return instruction + end_call_instructions
    else:
        # Use default instruction with agent name if provided
        if agent_name:
            return f"You are {agent_name}, a helpful AI voice assistant.\n\n{DEFAULT_SYSTEM_INSTRUCTION}"
        return DEFAULT_SYSTEM_INSTRUCTION


async def run_voice_bot(
    websocket_client,
    agent_prompt: str = None,
    agent_name: str = None,
    user_id: int = None,
    agent_id: int = None,
    db_session= None,
    current_conversation_id: int = None,
):
    """
    Run voice bot with agent-specific configuration

    Args:
        websocket_client: WebSocket connection
        agent_prompt: Custom prompt from the Agent model
        agent_name: Name of the agent
        user_id: ID of the user
        agent_id: ID of the agent
        db_session: Database session for saving data
        current_conversation_id: Active conversation ID
    """
    transcription_buffer = []
    conversation_ended = False  # Flag to track if conversation has ended
    BATCH_SIZE = 10  # Save transcriptions every 10 messages to prevent memory buildup

    # Create dynamic system instruction with agent's custom prompt
    SYSTEM_INSTRUCTION = create_voice_system_instruction(agent_prompt, agent_name)

    logger.info(f"Voice bot starting with agent '{agent_name}' (ID: {agent_id}) for user {user_id}")
    logger.debug(f"System instruction: {SYSTEM_INSTRUCTION[:200]}...")

    async def save_transcription_batch(batch_to_save):
        """Save a batch of transcriptions to database and clear memory"""
        if not batch_to_save:
            return

        logger.info(f"Saving batch of {len(batch_to_save)} transcriptions to database...")

        try:
            from models import Conversation, Transcription
            from sqlalchemy import select
            from datetime import datetime

            if not db_session or not db_session.is_active:
                logger.error("Database session is not active, cannot save transcriptions")
                return

            result = await db_session.execute(
                select(Conversation).filter(Conversation.uuid == current_conversation_id)
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                logger.error(f"Conversation {current_conversation_id} not found")
                return

            transcriptions_to_add = []
            for trans_data in batch_to_save:
                transcription = Transcription(
                    conversation_id=conversation.id,
                    text=trans_data["text"],
                    speaker=trans_data["speaker"],
                    timestamp=trans_data["timestamp"] or datetime.utcnow()
                )
                transcriptions_to_add.append(transcription)

            db_session.add_all(transcriptions_to_add)
            await db_session.commit()

            # CRITICAL: Expunge all objects to prevent memory leak
            db_session.expunge_all()

            # Clear the batch data from memory
            batch_to_save.clear()
            transcriptions_to_add.clear()

            logger.info(f"Successfully saved and cleared {len(transcriptions_to_add)} transcriptions from memory")

        except Exception as e:
            logger.error(f"Failed to save transcription batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                if db_session and db_session.is_active:
                    await db_session.rollback()
            except:
                pass

    async def save_all_transcriptions():
        """Save remaining buffered transcriptions to database using a fresh session"""
        if not transcription_buffer:
            logger.info("No transcriptions to save")
            return

        logger.info(f"Saving final {len(transcription_buffer)} transcriptions to database...")
        
        try:
            from models import Conversation, Transcription
            from sqlalchemy import select
            from datetime import datetime
            from main import AsyncSessionLocal  # Import your session factory
            
            # Create a fresh database session for final save
            async with AsyncSessionLocal() as fresh_session:
                result = await fresh_session.execute(
                    select(Conversation).filter(Conversation.uuid == current_conversation_id)
                )
                conversation = result.scalar_one_or_none()

                if not conversation:
                    logger.error(f"Conversation {current_conversation_id} not found")
                    transcription_buffer.clear()
                    return

                transcriptions_to_add = []
                for trans_data in transcription_buffer:
                    transcription = Transcription(
                        conversation_id=conversation.id,
                        text=trans_data["text"],
                        speaker=trans_data["speaker"],
                        timestamp=trans_data["timestamp"] or datetime.utcnow()
                    )
                    transcriptions_to_add.append(transcription)

                fresh_session.add_all(transcriptions_to_add)
                await fresh_session.commit()
                
                logger.info(f"Successfully saved final {len(transcriptions_to_add)} transcriptions")
                
        except Exception as e:
            logger.error(f"Failed to save final transcriptions: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            transcription_buffer.clear()


    async def extract_and_email_datapoints():
        """Extract datapoints and send email - separate from transcription save"""
        if not db_session or not db_session.is_active:
            logger.error("Database session is not active, cannot extract datapoints")
            return

        logger.info(f"Extracting datapoints for conversation {current_conversation_id}")
        transcription_list = []
        try:
            from models import Conversation, Transcription
            from sqlalchemy import select

            # Get conversation
            result = await db_session.execute(
                select(Conversation).filter(Conversation.uuid == current_conversation_id)
            )
            conversation = result.scalar_one_or_none()

            if conversation:
                # Get transcriptions using conversation_id (not uuid)
                trans_result = await db_session.execute(
                    select(Transcription)
                    .filter(Transcription.conversation_id == conversation.id)
                    .order_by(Transcription.timestamp)
                )
                transcriptions = trans_result.scalars().all()

                transcription_list = [
                    {
                        "timestamp": t.timestamp.isoformat(),
                        "speaker": t.speaker,
                        "text": t.text
                    }
                    for t in transcriptions
                ]

                # CRITICAL: Expunge to free memory from ORM objects
                db_session.expunge_all()

            # Extract datapoints
            datapoints = await extract_datapoints_from_conversation(
                conversation_uuid=current_conversation_id,
                db=db_session,
                agent_prompt=agent_prompt
            )
            logger.info(f"Datapoints extracted successfully: {datapoints.get('conversation_summary', 'N/A')}")

            # Send email
            await send_call_summary_email(
                recipient_email='bill@aijoe.ai',
                transcription=transcription_list,
                datapoints=datapoints,
                agent_name=agent_name,
                conversation_uuid=current_conversation_id
            )

            # Clear transcription list from memory
            transcription_list.clear()

        except Exception as e:
            logger.error(f"Failed to extract datapoints or send email: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Ensure cleanup even on error
            if transcription_list:
                transcription_list.clear()


    async def cleanup_resources():
        """Cleanup in-memory buffers and other resources to avoid leaks."""
        nonlocal transcription_buffer

        try:
            # Clear buffers
            transcription_buffer.clear()

            # Expunge all remaining ORM objects from session
            if db_session and db_session.is_active:
                db_session.expunge_all()

            logger.info("Transcription buffer and session objects cleared from memory")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def end_conversation_handler(params: FunctionCallParams):
        """Handle conversation end and extract datapoints"""
        nonlocal conversation_ended

        if conversation_ended:
            logger.info("Conversation already ended, skipping duplicate end")
            return

        # Lock immediately so disconnect handler won't run in parallel
        conversation_ended = True

        response = "Thank you for your time today! Feel free to reach out whenever you're ready. Have a great day!"

        await params.llm.push_frame(TTSSpeakFrame(response))
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"status": "ok"}, properties=FunctionCallResultProperties(run_llm=False)
        )

        await save_all_transcriptions()
        await extract_and_email_datapoints()
        await cleanup_resources()



    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )


    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-2.5-flash-native-audio-preview-09-2025",
        voice_id="Aoede",
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools_schema,
        transcribe_model_audio=True,
        transcribe_user_audio=True,
        params=InputParams(
            temperature=0.7,
            modalities=GeminiModalities.TEXT if agent_id == 3 else GeminiModalities.AUDIO,
            language=Language.EN_US,
        ),
    )
    logger.info(f"GeminiLiveLLMService initialized with model: gemini-2.5-flash-native-audio-preview-09-2025")

    # Replace the current tts initialization with:
    if agent_id == 3:
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JqGxQpW2LXaAdDhV6cHT"),
            model="eleven_flash_v2_5",
            params=ElevenLabsTTSService.InputParams(
                language=Language.EN,
                stability=0.7,
                similarity_boost=0.8,
                style=0.5,
                use_speaker_boost=True,
                speed=1.1,
            ),
        )
    else:
        tts = None  # No TTS for other agents

    # Replace the current pipeline definition with:

    llm.register_function("end_conversation", end_conversation_handler) 
    messages=[
        {
          "role": "user",
          "content": "Please introduce yourself briefly."
        }
    ]
    context = OpenAILLMContext(messages)
    
    from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

    context_aggregator = llm.create_context_aggregator(
        context,
    )
    transcript = TranscriptProcessor()
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    if agent_id == 3:
        pipeline = Pipeline([
            ws_transport.input(),
            rtvi,
            context_aggregator.user(),
            transcript.user(),  
            llm,
            tts,
            transcript.assistant(),
            ws_transport.output(),
            context_aggregator.assistant(),
        ])
    else:
        pipeline = Pipeline([
            ws_transport.input(),
            rtvi,
            context_aggregator.user(),
            transcript.user(),  
            llm,
            transcript.assistant(),
            ws_transport.output(),
            context_aggregator.assistant(),
        ])
        
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            interruption_strategy=MinWordsInterruptionStrategy(min_words=2),
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"Voice bot ready - Agent: {agent_name} (User: {user_id})")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected to agent {agent_name} (ID: {agent_id})")
        await task.queue_frames([LLMRunFrame()])

    @llm.event_handler("on_connection_established")
    async def on_llm_connected(service):
        logger.info(f"✓ Gemini Live connection ESTABLISHED for agent {agent_name}")

    @llm.event_handler("on_connection_lost")
    async def on_llm_disconnected(service):
        logger.error(f"✗ Gemini Live connection LOST for agent {agent_name}")

    @llm.event_handler("on_error")
    async def on_llm_error(service, error):
        logger.error(f"✗ Gemini Live ERROR for agent {agent_name}: {error}")

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        """Collect transcriptions in memory and save in batches to prevent memory leak"""
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"

                # Store in memory
                transcription_buffer.append({
                    "text": msg.content,
                    "speaker": msg.role,
                    "timestamp": msg.timestamp
                })

                logger.info(f"Transcript buffered: {line}")

                # Save in batches to prevent memory buildup
                if len(transcription_buffer) >= BATCH_SIZE:
                    logger.info(f"Buffer reached {BATCH_SIZE} items, saving batch...")
                    await save_transcription_batch(transcription_buffer.copy())
                    transcription_buffer.clear()

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Handle client disconnect - save transcriptions and extract datapoints"""
        nonlocal conversation_ended

        if conversation_ended:
            logger.info("Conversation already ended, skipping disconnect processing")
            return

        # Lock immediately
        conversation_ended = True

        logger.info(f"Client disconnected from agent {agent_name} (ID: {agent_id})")

        await save_all_transcriptions()
        await extract_and_email_datapoints()
        await cleanup_resources()

    async def run_voice_bot_webrtc(
        user_id: int,
        agent_id: int,
        conversation_uuid: str,
        agent_prompt: str,
        agent_name: str,
    ) -> dict:
        """
        Start a local WebRTC voice session (browser mic + speaker).
        Returns connection details the browser can use.
        """
        # Transport: local WebRTC (browser)
        transport = SmallWebRTCTransport(
            room_name=f"interview-{conversation_uuid}",
            audio_in_enabled=True,
            audio_out_enabled=True,
        )

        # Reuse your existing pipeline builder, but swap in the transport.
        # This assumes your existing run_voice_bot() builds a pipeline around a transport variable.
        # If it doesn't, we'll adjust in the next step.
        await run_voice_bot(
            websocket=None,  # not used in WebRTC mode
            user_id=user_id,
            agent_id=agent_id,
            current_conversation_id=conversation_uuid,
            agent_prompt=agent_prompt,
            agent_name=agent_name,
            transport_override=transport,   # <-- we’ll add this param next if needed
        )

        # SmallWebRTCTransport exposes join info; return it for the client.
        return transport.get_connection_info()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def run_human_voice_demo(
    websocket_client,
    persona,  # Persona object from persona_loader
    session_id: str,
):
    """
    Run a simplified voice demo for human-to-agent conversations.

    Args:
        websocket_client: WebSocket connection
        persona: Persona object with generate_system_prompt() method
        session_id: Ephemeral session identifier for logging
    """
    # Generate system instruction from persona
    SYSTEM_INSTRUCTION = persona.generate_system_prompt()

    # Add end conversation handling
    SYSTEM_INSTRUCTION += """

ENDING THE CALL:
You have access to an `end_conversation` function that disconnects the call. You MUST call this function to end the call properly.

When to call `end_conversation`:
- After the customer says goodbye, thank you, or indicates they're done (e.g., "bye", "thanks, that's all", "have a good day", "I'm all set")
- After YOU have said your brief farewell

How to end the call:
1. Say a brief, warm goodbye (e.g., "Have a great day!" or "Thank you for calling, goodbye!")
2. IMMEDIATELY call the `end_conversation` function - do NOT wait for another response

IMPORTANT: Once goodbyes have been exchanged, call `end_conversation` right away. Do not continue talking or ask additional questions after the customer has said goodbye.
"""

    logger.info(f"Human voice demo starting with persona '{persona.name}' (session: {session_id})")
    logger.debug(f"System instruction: {SYSTEM_INSTRUCTION[:200]}...")

    # Task reference for the end_conversation handler
    task_ref = {"task": None}

    async def end_conversation_handler(params: FunctionCallParams):
        """Handle conversation end - closes the connection"""
        logger.info(f"end_conversation called for session {session_id}")

        # Return result first to acknowledge the function call
        await params.result_callback(
            {"status": "ok"}, properties=FunctionCallResultProperties(run_llm=False)
        )

        # Cancel the task to stop the pipeline and close websocket
        if task_ref["task"]:
            logger.info(f"Cancelling task for session {session_id}")
            await task_ref["task"].cancel()

        logger.info(f"Human voice demo ended for session {session_id}")

    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-2.5-flash-native-audio-preview-09-2025",
        voice_id="Aoede",
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools_schema,
        transcribe_model_audio=True,
        transcribe_user_audio=True,
        params=InputParams(
            temperature=0.7,
            modalities=GeminiModalities.AUDIO,
            language=Language.EN_US,
        ),
    )
    logger.info(f"GeminiLiveLLMService initialized for human demo")

    llm.register_function("end_conversation", end_conversation_handler)

    messages = [
        {
            "role": "user",
            "content": "Please introduce yourself briefly."
        }
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    transcript = TranscriptProcessor()
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline([
        ws_transport.input(),
        rtvi,
        context_aggregator.user(),
        transcript.user(),
        llm,
        transcript.assistant(),
        ws_transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            interruption_strategy=MinWordsInterruptionStrategy(min_words=2),
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Store task reference for the end_conversation handler
    task_ref["task"] = task

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"Human voice demo ready - Persona: {persona.name}")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected to human demo (persona: {persona.name})")
        await task.queue_frames([LLMRunFrame()])

    @llm.event_handler("on_connection_established")
    async def on_llm_connected(service):
        logger.info(f"✓ Gemini Live connection ESTABLISHED for human demo")

    @llm.event_handler("on_connection_lost")
    async def on_llm_disconnected(service):
        logger.error(f"✗ Gemini Live connection LOST for human demo")

    @llm.event_handler("on_error")
    async def on_llm_error(service, error):
        logger.error(f"✗ Gemini Live ERROR for human demo: {error}")

    async def delayed_disconnect():
        """Wait for audio to finish playing, then disconnect"""
        await asyncio.sleep(4)  # Give time for goodbye audio to play
        logger.info(f"[Demo] Auto-disconnecting after goodbye exchange")
        if task_ref["task"]:
            await task_ref["task"].cancel()

    # Track if we're in goodbye mode
    goodbye_state = {"user_said_bye": False, "ended": False}

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        """Log transcriptions and detect goodbye to auto-disconnect"""
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"
                logger.info(f"[Demo Transcript] {line}")

                content_lower = msg.content.lower().strip()

                # Detect user saying goodbye
                if msg.role == "user":
                    goodbye_words = [
                        "bye", "goodbye", "talk to you later", "gotta go", "have to go",
                        "i'm all set", "that's all", "thanks, bye", "call you back",
                        "call you right back", "got to go", "need to go", "heading out",
                        "that's it for now", "i'm done", "all set"
                    ]
                    if any(word in content_lower for word in goodbye_words):
                        goodbye_state["user_said_bye"] = True
                        logger.info(f"[Demo] User said goodbye, waiting for agent response")

                # If user said bye and now agent responds with goodbye, disconnect
                if msg.role == "assistant" and goodbye_state["user_said_bye"] and not goodbye_state["ended"]:
                    agent_goodbye_words = [
                        "goodbye", "bye", "have a great day", "have a wonderful day",
                        "take care", "talk to you later", "thank you for calling",
                        "thanks for calling", "have a good day", "have a nice day",
                        "call back", "reach out", "anytime"
                    ]
                    if any(word in content_lower for word in agent_goodbye_words):
                        goodbye_state["ended"] = True
                        logger.info(f"[Demo] Agent said goodbye after user - scheduling disconnect")
                        # Schedule disconnect after a short delay to let the audio finish
                        asyncio.create_task(delayed_disconnect())

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected from human demo (persona: {persona.name})")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)