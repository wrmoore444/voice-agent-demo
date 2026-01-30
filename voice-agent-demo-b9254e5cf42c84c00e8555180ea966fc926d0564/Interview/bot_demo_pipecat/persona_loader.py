"""
=============================================================================
PERSONA LOADER - Loads persona JSON files and generates LLM system prompts
=============================================================================

This module is the bridge between persona JSON configuration files and the
LLM system prompts that drive bot behavior.

HOW IT FITS IN THE SYSTEM:
--------------------------
1. Persona JSON files define bot personalities, goals, and conversation stages
2. This module loads those JSON files into Persona objects
3. generate_system_prompt() converts a Persona into text for the LLM
4. dual_bot_service.py uses these prompts when calling GoogleLLMService

KEY COMPONENTS:
---------------
- Persona dataclass: Holds all persona configuration
- generate_system_prompt(): Builds the complete LLM system prompt
- CRITICAL RULES: Hardcoded rules that apply to ALL personas (lines 76-84)

TO MODIFY BOT BEHAVIOR:
-----------------------
- Persona-specific (personality, stages, etc.): Edit the JSON files
- Rules for ALL personas: Edit CRITICAL RULES in generate_system_prompt()

PERSONA JSON LOCATION:
----------------------
JSON files are stored in: Interview/bot_demo/personas/
Naming convention: alice_<role>.json, bob_<role>.json
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to persona JSON files (relative to this file's location)
# Points to: Interview/bot_demo/personas/
PERSONAS_DIR = Path(__file__).parent.parent / "bot_demo" / "personas"


# =============================================================================
# PERSONA DATA CLASS
# =============================================================================

@dataclass
class Persona:
    """
    A loaded persona with its stages and configuration.

    This dataclass holds all the information extracted from a persona JSON file.
    The key method is generate_system_prompt(), which converts this data into
    the text that gets sent to the LLM as its system instructions.

    ATTRIBUTES (from JSON):
    -----------------------
    - name: The bot's name (e.g., "Alice", "Bob Patterson")
    - role: Job title/role (e.g., "Customer Service Representative")
    - personality: Description of how the bot should behave
    - voice_notes: Speaking style guidance
    - goal: What the bot is trying to accomplish
    - restrictions: Things the bot should NOT do
    - stages: List of conversation stages with objectives and data points
    - company_name: The company the bot works for
    - raw_data: The complete original JSON (for advanced use)

    JSON STRUCTURE EXPECTED:
    ------------------------
    {
        "UseCase": {
            "Assistant": {
                "Name": "...",
                "Role": "...",
                "Personality": "...",
                "VoiceNotes": "...",
                "Goal": "...",
                "Restrictions": "..."
            },
            "Company": {
                "CompanyName": "..."
            }
        },
        "Stages": [
            {
                "StageName": "Greeting",
                "Objectives": ["...", "..."],
                "DataPoints": [{"DatapointName": "...", "DatapointDescription": "..."}],
                "ExamplePhrases": ["...", "..."],
                "CompletionCriteria": "..."
            },
            ...
        ]
    }
    """
    name: str
    role: str
    personality: str
    voice_notes: str
    goal: str
    restrictions: str
    stages: List[Dict[str, Any]]
    company_name: str
    raw_data: Dict[str, Any]

    def generate_system_prompt(self) -> str:
        """
        Generate a complete system prompt from the persona.

        This method builds the text that gets sent to the LLM as its
        "system" message - the instructions that define how it should behave.

        PROMPT STRUCTURE:
        -----------------
        1. Identity: "You are {name}, {role}"
        2. PERSONALITY: How to behave
        3. GOAL: What to accomplish
        4. VOICE & STYLE: Speaking manner
        5. RESTRICTIONS: What NOT to do
        6. CONVERSATION STAGES: Step-by-step guide (from _format_stages)
        7. CRITICAL RULES: Universal rules for all personas

        CRITICAL RULES (lines 76-84):
        -----------------------------
        These rules are HARDCODED here, not in the JSON files. They apply
        to ALL personas and control fundamental behaviors:
        - Response length limits
        - Listening/acknowledgment behavior
        - Stage following instructions
        - Conversational style requirements

        TO MODIFY CRITICAL RULES:
        Edit the text in the prompt string below (lines 76-84).
        Changes will affect ALL personas.

        Returns:
            Complete system prompt string ready for LLM
        """
        # Build the formatted stages section
        stages_text = self._format_stages()

        # Assemble the complete prompt
        # NOTE: The CRITICAL RULES section (lines 76-84) is hardcoded here
        # and applies to ALL personas. Edit here to change universal behavior.
        prompt = f"""You are "{self.name}", {self.role}

PERSONALITY:
{self.personality}

GOAL:
{self.goal}

VOICE & STYLE:
{self.voice_notes}

RESTRICTIONS:
{self.restrictions}

{stages_text}

CRITICAL RULES:
- KEEP RESPONSES VERY SHORT: 1-2 sentences maximum. This is a real-time conversation.
- LISTEN CAREFULLY: The customer may provide information BEFORE you ask for it. Recognize and acknowledge this.
- NEVER ask for information the customer has already provided. If they said "Seattle for my brother's wedding", you already have destination AND occasion.
- Follow the stages in order, but skip questions when data points have already been captured from what the customer said.
- Use the example phrases as inspiration, but adapt based on what information you already have.
- React authentically to what the other person says.
- Do not announce stage names or transitions out loud.
- Never give long explanations. Be conversational, not verbose.
"""
        return prompt

    def _format_stages(self) -> str:
        """
        Format conversation stages into readable text for the prompt.

        Converts the stages list from JSON into a formatted text block
        that the LLM can understand and follow.

        OUTPUT FORMAT:
        --------------
        CONVERSATION STAGES:
        ========================================

        Stage 1: Greeting
        ------------------------------
        Objectives:
          - Welcome the customer
          - Establish rapport

        Data to collect:
          - CustomerName: The customer's name

        Example phrases:
          - "Welcome to First National Bank!"
          - "How can I help you today?"

        Completion: Move to Stage 2 when greeting is complete.

        Stage 2: Problem Discovery
        ...

        NOTE: Only first 2 example phrases are included per stage
        to keep the prompt concise.

        Returns:
            Formatted stages text block
        """
        lines = ["CONVERSATION STAGES:", "=" * 40]

        for i, stage in enumerate(self.stages, 1):
            # Stage header
            lines.append(f"\nStage {i}: {stage['StageName']}")
            lines.append("-" * 30)

            # Objectives - what the bot should accomplish in this stage
            lines.append("Objectives:")
            for obj in stage.get('Objectives', []):
                lines.append(f"  - {obj}")

            # Data points - information to collect from the customer
            lines.append("\nData to collect:")
            for dp in stage.get('DataPoints', []):
                lines.append(f"  - {dp['DatapointName']}: {dp['DatapointDescription']}")

            # Example phrases - sample things the bot might say
            # Limited to 2 to keep prompt concise
            lines.append("\nExample phrases:")
            for phrase in stage.get('ExamplePhrases', [])[:2]:
                lines.append(f'  - "{phrase}"')

            # Completion criteria - when to move to next stage
            lines.append(f"\nCompletion: {stage.get('CompletionCriteria', 'Move to next stage when objectives are met.')}")

        return "\n".join(lines)


# =============================================================================
# PERSONA LOADING FUNCTIONS
# =============================================================================

def load_persona(filename: str) -> Optional[Persona]:
    """
    Load a persona from a JSON file.

    Reads a persona JSON file and converts it into a Persona object
    that can be used to generate system prompts.

    Args:
        filename: Either a full absolute path to a JSON file, or just
                  the filename (e.g., "alice_bank_teller.json") which
                  will be looked up in PERSONAS_DIR.

    Returns:
        Persona object with all configuration loaded

    Raises:
        FileNotFoundError: If the persona file doesn't exist

    EXAMPLE USAGE:
    --------------
        # By filename (looks in PERSONAS_DIR)
        persona = load_persona("alice_bank_teller.json")

        # By full path
        persona = load_persona("/path/to/custom_persona.json")

        # Generate prompt for LLM
        system_prompt = persona.generate_system_prompt()
    """
    # Check if it's a full path or just filename
    if os.path.isabs(filename):
        filepath = Path(filename)
    else:
        filepath = PERSONAS_DIR / filename

    # Add .json extension if not present
    if not filepath.suffix:
        filepath = filepath.with_suffix('.json')

    if not filepath.exists():
        raise FileNotFoundError(f"Persona file not found: {filepath}")

    # Load and parse JSON
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract nested structures from JSON
    use_case = data.get('UseCase', {})
    assistant = use_case.get('Assistant', {})
    company = use_case.get('Company', {})

    # Build and return Persona object
    return Persona(
        name=assistant.get('Name', 'Unknown'),
        role=assistant.get('Role', ''),
        personality=assistant.get('Personality', ''),
        voice_notes=assistant.get('VoiceNotes', ''),
        goal=assistant.get('Goal', ''),
        restrictions=assistant.get('Restrictions', ''),
        stages=data.get('Stages', []),
        company_name=company.get('CompanyName', ''),
        raw_data=data  # Keep original for advanced use
    )


def list_personas() -> Dict[str, List[str]]:
    """
    List all available persona files, grouped by type (alice/bob).

    Scans the PERSONAS_DIR for JSON files and categorizes them based
    on their filename prefix. Used by the API to show available options.

    Returns:
        Dictionary with "alice" and "bob" keys, each containing a list
        of persona filenames (without .json extension).

    EXAMPLE RETURN:
    ---------------
        {
            "alice": ["alice_bank_teller", "alice_insurance_agent", "alice_travel_agent"],
            "bob": ["bob_bank_upset_customer", "bob_insurance_frustrated_claimant", ...]
        }
    """
    if not PERSONAS_DIR.exists():
        return {"alice": [], "bob": []}

    personas = {"alice": [], "bob": []}

    for file in PERSONAS_DIR.glob("*.json"):
        name = file.stem  # Filename without extension
        if name.startswith("alice_"):
            personas["alice"].append(name)
        elif name.startswith("bob_"):
            personas["bob"].append(name)

    return personas


def get_default_personas() -> tuple:
    """
    Get the default persona filenames.

    Returns the personas that will be used if none are specified
    when starting a conversation.

    Returns:
        Tuple of (alice_persona_filename, bob_persona_filename)

    TO CHANGE DEFAULTS:
    -------------------
    Edit the return values below to use different default personas.
    """
    return ("alice_insurance_agent.json", "bob_insurance_frustrated_claimant.json")
