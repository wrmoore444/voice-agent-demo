"""
Persona Testing Module - EXPERIMENTAL

Runs automated conversations between a customer service agent and all
matching customer personas, then analyzes the results to suggest improvements.

This module is experimental and can be safely removed without affecting
the main bot demo functionality.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from .persona_loader import load_persona, Persona, PERSONAS_DIR


# Output directory for test results
RESULTS_DIR = Path(__file__).parent / "test_results"


@dataclass
class ConversationMessage:
    """A single message in a test conversation."""
    speaker: str
    text: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp
        }


@dataclass
class ConversationAnalysis:
    """Analysis of a single conversation."""
    objectives_met: Dict[str, bool] = field(default_factory=dict)
    data_points_captured: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    overall_rating: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives_met": self.objectives_met,
            "data_points_captured": self.data_points_captured,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "overall_rating": self.overall_rating
        }


@dataclass
class ConversationResult:
    """Result of a single test conversation."""
    customer_persona: str
    messages: List[ConversationMessage] = field(default_factory=list)
    analysis: Optional[ConversationAnalysis] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_persona": self.customer_persona,
            "messages": [m.to_dict() for m in self.messages],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "error": self.error
        }


@dataclass
class TestRunResult:
    """Complete result of a test run."""
    agent_persona: str
    test_run_id: str
    started_at: str
    completed_at: Optional[str] = None
    conversations: List[ConversationResult] = field(default_factory=list)
    overall_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_persona": self.agent_persona,
            "test_run_id": self.test_run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "conversations": [c.to_dict() for c in self.conversations],
            "overall_analysis": self.overall_analysis
        }


def get_industry_from_persona(persona_filename: str) -> Optional[str]:
    """Extract industry from persona filename."""
    # Format: alice_<industry>_<role>.json or bob_<industry>_<type>.json
    name = persona_filename.replace('.json', '')
    parts = name.split('_')
    if len(parts) >= 2:
        # alice_travel_agent -> travel
        # bob_insurance_frustrated_claimant -> insurance
        return parts[1]
    return None


def find_customer_personas_for_agent(agent_filename: str) -> List[str]:
    """Find all customer (bob) personas matching the agent's industry."""
    industry = get_industry_from_persona(agent_filename)
    if not industry:
        logger.warning(f"Could not determine industry from {agent_filename}")
        return []

    customer_personas = []
    for file in PERSONAS_DIR.glob("bob_*.json"):
        if industry in file.name:
            customer_personas.append(file.name)

    logger.info(f"Found {len(customer_personas)} customer personas for industry '{industry}'")
    return sorted(customer_personas)


async def run_single_conversation(
    agent_persona: Persona,
    customer_persona: Persona,
    max_turns: int = 50
) -> List[ConversationMessage]:
    """Run a single conversation between agent and customer."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)

    messages: List[ConversationMessage] = []

    # Create chat sessions
    agent_chat = _create_chat(agent_persona, is_starter=True)
    customer_chat = _create_chat(customer_persona, is_starter=False)

    # Agent starts
    initial_prompt = "Begin the conversation according to your Stage 1 instructions. Keep your response to 1-2 short sentences."
    agent_response = await _generate_response(agent_chat, initial_prompt)

    if not agent_response:
        raise RuntimeError("Agent failed to generate initial response")

    messages.append(ConversationMessage(
        speaker=agent_persona.name,
        text=agent_response,
        timestamp=datetime.utcnow().isoformat()
    ))

    # Conversation loop
    turn_count = 0
    closing_exchange = False
    conversation_ended = False

    while turn_count < max_turns and not conversation_ended:
        turn_count += 1

        # Customer responds
        context = _get_context(messages, customer_persona.name)
        if closing_exchange:
            context += "\n\n(The conversation is wrapping up. Give a brief closing response.)"

        customer_response = await _generate_response(customer_chat, context)
        if not customer_response:
            break

        messages.append(ConversationMessage(
            speaker=customer_persona.name,
            text=customer_response,
            timestamp=datetime.utcnow().isoformat()
        ))

        if _is_conversation_ending(customer_response):
            if closing_exchange:
                conversation_ended = True
                continue
            else:
                closing_exchange = True

        turn_count += 1
        if turn_count >= max_turns:
            break

        # Agent responds
        context = _get_context(messages, agent_persona.name)
        if closing_exchange:
            context += "\n\n(The customer is wrapping up. Give a brief, professional closing.)"

        agent_response = await _generate_response(agent_chat, context)
        if not agent_response:
            break

        messages.append(ConversationMessage(
            speaker=agent_persona.name,
            text=agent_response,
            timestamp=datetime.utcnow().isoformat()
        ))

        if _is_conversation_ending(agent_response):
            if closing_exchange:
                conversation_ended = True
            else:
                closing_exchange = True

    return messages


def _create_chat(persona: Persona, is_starter: bool):
    """Create a Gemini chat session for a bot."""
    import google.generativeai as genai

    system_prompt = persona.generate_system_prompt()
    if is_starter:
        system_prompt += "\n\nYou are starting this conversation. Begin with your Stage 1 greeting."

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.8,
            max_output_tokens=75,
        ),
    )
    return model.start_chat()


async def _generate_response(chat, prompt: str) -> Optional[str]:
    """Generate a response from a chat session."""
    try:
        response = await asyncio.to_thread(chat.send_message, prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None


def _get_context(messages: List[ConversationMessage], for_speaker: str, last_n: int = 5) -> str:
    """Get recent conversation context."""
    recent = messages[-last_n:]
    if not recent:
        return "Continue the conversation. Reply with 1-2 short sentences only."

    lines = []
    for msg in recent:
        if msg.speaker != for_speaker:
            lines.append(f"{msg.speaker}: {msg.text}")

    context = "\n".join(lines) if lines else "Continue the conversation."
    return f"{context}\n\n(Reply with 1-2 short sentences only. Be conversational.)"


def _is_conversation_ending(text: str) -> bool:
    """Check if the response indicates conversation is ending."""
    text_lower = text.lower()
    farewell_phrases = [
        "goodbye", "good bye", "bye", "have a good day", "have a great day",
        "take care", "talk to you later", "thanks for your help",
        "thank you for your help", "that's all", "that's everything",
        "nothing else", "no, that's it", "no that's it"
    ]
    return any(phrase in text_lower for phrase in farewell_phrases)


async def analyze_conversation(
    agent_persona: Persona,
    messages: List[ConversationMessage]
) -> ConversationAnalysis:
    """Analyze a conversation against the agent's stages."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    # Build conversation transcript
    transcript = "\n".join([f"{m.speaker}: {m.text}" for m in messages])

    # Build stages info for analysis
    stages_info = _format_stages_for_analysis(agent_persona)

    analysis_prompt = f"""Analyze this customer service conversation and evaluate the agent's performance.

AGENT: {agent_persona.name}
ROLE: {agent_persona.role}
GOAL: {agent_persona.goal}

STAGES AND OBJECTIVES:
{stages_info}

CONVERSATION TRANSCRIPT:
{transcript}

Analyze the conversation and provide:

1. OBJECTIVES_MET: For each stage objective, was it achieved? (true/false with brief reason)

2. DATA_POINTS_CAPTURED: For each data point the agent should collect, was it captured? (true/false)

3. ISSUES: List specific problems you observed:
   - Did the agent ask for information already provided?
   - Were responses too long or verbose?
   - Did the agent miss cues from the customer?
   - Were transitions between stages awkward?
   - Did the agent fail to address customer concerns?

4. SUGGESTIONS: Specific improvements for the agent's persona/stages file:
   - What phrases or approaches worked well?
   - What should be changed in the example phrases?
   - Are there missing objectives or data points?
   - Should completion criteria be adjusted?

5. OVERALL_RATING: Rate as EXCELLENT, GOOD, NEEDS_IMPROVEMENT, or POOR with brief explanation.

Respond in JSON format:
{{
  "objectives_met": {{"Stage - Objective": {{"met": true/false, "reason": "..."}}, ...}},
  "data_points_captured": {{"DataPointName": true/false, ...}},
  "issues": ["issue 1", "issue 2", ...],
  "suggestions": ["suggestion 1", "suggestion 2", ...],
  "overall_rating": "RATING: explanation"
}}"""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        response = await asyncio.to_thread(
            model.generate_content, analysis_prompt
        )

        # Parse JSON from response
        text = response.text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)

        return ConversationAnalysis(
            objectives_met=data.get("objectives_met", {}),
            data_points_captured=data.get("data_points_captured", {}),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            overall_rating=data.get("overall_rating", "")
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return ConversationAnalysis(
            issues=[f"Analysis failed: {str(e)}"],
            overall_rating="UNKNOWN: Analysis error"
        )


def _format_stages_for_analysis(persona: Persona) -> str:
    """Format stages for the analysis prompt."""
    lines = []
    for stage in persona.stages:
        lines.append(f"\nStage: {stage['StageName']}")
        lines.append("Objectives:")
        for obj in stage.get('Objectives', []):
            lines.append(f"  - {obj}")
        lines.append("Data Points to Collect:")
        for dp in stage.get('DataPoints', []):
            lines.append(f"  - {dp['DatapointName']}: {dp['DatapointDescription']}")
        lines.append(f"Completion: {stage.get('CompletionCriteria', 'N/A')}")
    return "\n".join(lines)


async def generate_overall_analysis(
    agent_persona: Persona,
    conversation_results: List[ConversationResult]
) -> Dict[str, Any]:
    """Generate overall analysis across all conversations."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    # Compile all issues and suggestions
    all_issues = []
    all_suggestions = []
    ratings = []

    for result in conversation_results:
        if result.analysis:
            all_issues.extend(result.analysis.issues)
            all_suggestions.extend(result.analysis.suggestions)
            if result.analysis.overall_rating:
                ratings.append(f"{result.customer_persona}: {result.analysis.overall_rating}")

    summary_prompt = f"""Based on testing the customer service agent "{agent_persona.name}" with multiple customer types,
provide a consolidated improvement plan.

AGENT: {agent_persona.name}
ROLE: {agent_persona.role}

INDIVIDUAL RATINGS:
{chr(10).join(ratings)}

ALL ISSUES OBSERVED:
{chr(10).join(f"- {issue}" for issue in all_issues)}

ALL SUGGESTIONS:
{chr(10).join(f"- {s}" for s in all_suggestions)}

Provide:
1. COMMON_PATTERNS: Issues that appeared across multiple conversations
2. PRIORITY_IMPROVEMENTS: Top 5 changes to make to the agent's persona file, ranked by impact
3. STAGE_SPECIFIC_CHANGES: Specific edits recommended for each stage
4. OVERALL_ASSESSMENT: Summary of agent's strengths and weaknesses

Respond in JSON format."""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        response = await asyncio.to_thread(
            model.generate_content, summary_prompt
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        return json.loads(text)

    except Exception as e:
        logger.error(f"Overall analysis error: {e}")
        return {
            "error": str(e),
            "all_issues": all_issues,
            "all_suggestions": all_suggestions
        }


async def run_persona_test(agent_filename: str) -> TestRunResult:
    """
    Run a complete test of an agent against all matching customer personas.

    Args:
        agent_filename: The agent persona filename (e.g., "alice_travel_agent.json")

    Returns:
        TestRunResult with all conversations and analysis
    """
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_run_id = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    result = TestRunResult(
        agent_persona=agent_filename,
        test_run_id=test_run_id,
        started_at=datetime.utcnow().isoformat()
    )

    try:
        # Load agent persona
        agent_persona = load_persona(agent_filename)
        logger.info(f"Testing agent: {agent_persona.name}")

        # Find matching customer personas
        customer_files = find_customer_personas_for_agent(agent_filename)
        if not customer_files:
            raise ValueError(f"No customer personas found for {agent_filename}")

        logger.info(f"Will test against {len(customer_files)} customer personas")

        # Run each conversation
        for customer_file in customer_files:
            logger.info(f"Running conversation with {customer_file}...")

            conv_result = ConversationResult(customer_persona=customer_file)

            try:
                customer_persona = load_persona(customer_file)

                # Run the conversation
                messages = await run_single_conversation(
                    agent_persona=agent_persona,
                    customer_persona=customer_persona
                )
                conv_result.messages = messages

                # Analyze the conversation
                logger.info(f"Analyzing conversation with {customer_file}...")
                conv_result.analysis = await analyze_conversation(
                    agent_persona=agent_persona,
                    messages=messages
                )

            except Exception as e:
                logger.error(f"Error with {customer_file}: {e}")
                conv_result.error = str(e)

            result.conversations.append(conv_result)

        # Generate overall analysis
        logger.info("Generating overall analysis...")
        result.overall_analysis = await generate_overall_analysis(
            agent_persona=agent_persona,
            conversation_results=result.conversations
        )

        result.completed_at = datetime.utcnow().isoformat()

        # Save results
        output_file = RESULTS_DIR / f"{agent_filename.replace('.json', '')}_{test_run_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.exception(f"Test run failed: {e}")
        result.overall_analysis = {"error": str(e)}
        result.completed_at = datetime.utcnow().isoformat()

    return result


def list_test_results() -> List[Dict[str, str]]:
    """List all available test result files."""
    if not RESULTS_DIR.exists():
        return []

    results = []
    for file in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        results.append({
            "filename": file.name,
            "path": str(file),
            "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })
    return results


def load_test_result(filename: str) -> Optional[Dict[str, Any]]:
    """Load a specific test result file."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
