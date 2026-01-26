import os
import json
from loguru import logger
from typing import List, Dict, Optional
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCIFSS8c9-oxJ6i8d3yfYYcNscs-jgFFiE"))


async def extract_datapoints_from_conversation(
    conversation_uuid: str,
    db: AsyncSession,
    agent_prompt: Optional[str] = None
) -> Dict:
    """
    Extract structured datapoints from a conversation using Gemini AI.
    
    Args:
        conversation_uuid: UUID of the conversation to analyze
        db: Async database session
        agent_prompt: Custom agent prompt to understand context (optional)
    
    Returns:
        Dictionary containing extracted datapoints
    """

    # These are defined outside try so we can safely clean them in finally
    transcriptions = []
    transcript_text: Optional[str] = None
    datapoints: Dict = {}

    try:
        # Import models here to avoid circular imports
        from models import Conversation, Transcription

        # Fetch the conversation
        result = await db.execute(
            select(Conversation)
            .filter(Conversation.uuid == conversation_uuid)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            logger.error(f"Conversation {conversation_uuid} not found")
            return {}

        # Get all transcriptions for this conversation
        trans_result = await db.execute(
            select(Transcription)
            .filter(Transcription.conversation_id == conversation.id)
            .order_by(Transcription.timestamp)
        )
        transcriptions = trans_result.scalars().all()

        if not transcriptions:
            logger.warning(f"No transcriptions found for conversation {conversation_uuid}")
            return {}

        # Format the conversation transcript
        transcript_text = format_transcript(transcriptions)

        # Create extraction prompt based on agent's purpose
        extraction_prompt = create_extraction_prompt(transcript_text, agent_prompt)
        logger.debug("Extraction Prompt:\n" + extraction_prompt[:500])

        # Use Gemini to extract datapoints
        model = genai.GenerativeModel('gemini-2.5-flash')

        # NOTE: generate_content is sync; fine to call from async,
        # but if it becomes a bottleneck you may want to wrap it in a threadpool.
        response = model.generate_content(extraction_prompt)
        logger.debug("Gemini Response:\n" + response.text[:500])

        # Parse the response
        datapoints = parse_gemini_response(response.text)

        # Update the conversation with datapoints
        conversation.datapoints = json.dumps(datapoints, indent=2)

        # Commit the changes
        await db.commit()

        # CRITICAL: Expunge all objects to prevent memory leak from ORM objects
        db.expunge_all()

        logger.info(f"Successfully extracted and saved datapoints for conversation {conversation_uuid}")
        logger.debug(f"Datapoints: {datapoints}")

        return datapoints

    except Exception as e:
        logger.error(f"Error extracting datapoints: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            await db.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback failed: {rb_e}")
        return {}

    finally:
        # 🔻 Cleanup to avoid memory buildup
        try:
            if transcriptions:
                # Clear list of ORM objects so the big list can be GC'd
                transcriptions.clear()

            # Drop large string from memory
            transcript_text = None

            logger.info("Cleanup complete in extract_datapoints_from_conversation: transcription data cleared.")
        except Exception as ce:
            logger.error(f"Cleanup error in extract_datapoints_from_conversation: {ce}")


def format_transcript(transcriptions: List) -> str:
    """Format transcriptions into a readable conversation transcript"""
    transcript_lines = []

    for trans in transcriptions:
        speaker = trans.speaker.upper() if trans.speaker else "UNKNOWN"
        text = (trans.text or "").strip()
        transcript_lines.append(f"{speaker}: {text}")

    result = "\n".join(transcript_lines)

    # Optional micro-cleanup (locals get GC'd anyway, but this is explicit)
    transcript_lines.clear()

    return result


def create_extraction_prompt(transcript: str, agent_prompt: Optional[str] = None) -> str:
    """
    Create a prompt for Gemini to extract relevant datapoints from the conversation.
    """

    base_prompt = f"""You are a data extraction assistant. Analyze the following conversation transcript and extract all relevant datapoints, information, and insights.

{"AGENT'S ROLE/PURPOSE:" if agent_prompt else ""}
{agent_prompt if agent_prompt else ""}

CONVERSATION TRANSCRIPT:
{transcript}

Extract and return a JSON object with the following information:

1. **contact_information**: Any contact details mentioned (name, email, phone, company, etc.)
2. **key_topics**: Main topics discussed in the conversation
3. **user_needs**: What the user is looking for or interested in
4. **pain_points**: Any problems or challenges mentioned by the user
5. **next_steps**: Any agreed-upon follow-up actions or next steps
6. **important_dates**: Any dates, deadlines, or time-sensitive information mentioned
7. **product_interest**: Specific products, services, or features the user showed interest in
8. **budget_info**: Any budget-related information mentioned
9. **decision_timeline**: When the user plans to make a decision
10. **sentiment**: Overall sentiment of the conversation (positive/neutral/negative)
11. **conversation_summary**: A brief summary of the entire conversation (2-3 sentences)
12. **custom_fields**: Any other important information specific to this conversation

Return ONLY valid JSON. Do not include any markdown formatting, explanations, or text outside the JSON structure.

Example format:
{{
  "contact_information": {{
    "name": "John Doe",
    "email": "john@example.com",
    "company": "Acme Corp"
  }},
  "key_topics": ["product demo", "pricing", "implementation"],
  "user_needs": ["automated reporting", "team collaboration"],
  "pain_points": ["manual data entry", "lack of integration"],
  "next_steps": ["send pricing proposal", "schedule follow-up call"],
  "important_dates": ["meeting scheduled for next Tuesday"],
  "product_interest": ["Enterprise plan", "API access"],
  "budget_info": "10-15K annually",
  "decision_timeline": "end of quarter",
  "sentiment": "positive",
  "conversation_summary": "User is interested in our enterprise solution for their team of 50. They want to see pricing and schedule a demo.",
  "custom_fields": {{}}
}}

IMPORTANT: Return ONLY the JSON object, nothing else."""

    return base_prompt


def parse_gemini_response(response_text: str) -> Dict:
    """
    Parse Gemini's response and extract the JSON datapoints.
    Handles cases where the response might include markdown code blocks.
    """
    try:
        # Remove markdown code blocks if present
        cleaned_text = response_text.strip()

        # Remove ```json and ``` markers
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        cleaned_text = cleaned_text.strip()

        # Parse JSON
        datapoints = json.loads(cleaned_text)

        return datapoints

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.debug(f"Response text: {response_text}")

        # Return a basic structure if parsing fails
        return {
            "error": "Failed to parse response",
            "raw_response": response_text[:500],  # First 500 chars
            "conversation_summary": "Unable to extract structured data"
        }


async def get_conversation_datapoints(conversation_uuid: str, db_session: AsyncSession) -> Optional[Dict]:
    """
    Retrieve saved datapoints for a conversation.
    
    Args:
        conversation_uuid: UUID of the conversation
        db_session: Database session
    
    Returns:
        Dictionary of datapoints or None if not found
    """
    try:
        from models import Conversation

        result = await db_session.execute(
            select(Conversation).filter(Conversation.uuid == conversation_uuid)
        )
        conversation = result.scalar_one_or_none()
        await result.close()  # CRITICAL: Close result to return connection to pool

        if conversation and conversation.datapoints:
            return json.loads(conversation.datapoints)

        return None

    except Exception as e:
        logger.error(f"Error retrieving datapoints: {e}")
        return None