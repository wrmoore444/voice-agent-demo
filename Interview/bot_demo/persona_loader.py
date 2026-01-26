"""Loads persona JSON files and generates system prompts."""

import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

PERSONAS_DIR = Path(__file__).parent / "personas"


@dataclass
class Persona:
    """A loaded persona with its stages and configuration."""
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
        """Generate a complete system prompt from the persona."""

        # Build stages section
        stages_text = self._format_stages()

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

IMPORTANT RULES:
- Follow the stages in order. Do not skip stages.
- Collect the required data points before moving to the next stage.
- Use the example phrases as inspiration, but speak naturally.
- Keep responses SHORT - 1-3 sentences maximum.
- React authentically to what the other person says.
- Do not announce stage names or transitions out loud.
"""
        return prompt

    def _format_stages(self) -> str:
        """Format stages into readable text for the prompt."""
        lines = ["CONVERSATION STAGES:", "=" * 40]

        for i, stage in enumerate(self.stages, 1):
            lines.append(f"\nStage {i}: {stage['StageName']}")
            lines.append("-" * 30)

            # Objectives
            lines.append("Objectives:")
            for obj in stage.get('Objectives', []):
                lines.append(f"  - {obj}")

            # Data to collect
            lines.append("\nData to collect:")
            for dp in stage.get('DataPoints', []):
                lines.append(f"  - {dp['DatapointName']}: {dp['DatapointDescription']}")

            # Example phrases
            lines.append("\nExample phrases:")
            for phrase in stage.get('ExamplePhrases', [])[:2]:  # Limit to 2 examples
                lines.append(f'  - "{phrase}"')

            # Completion criteria
            lines.append(f"\nCompletion: {stage.get('CompletionCriteria', 'Move to next stage when objectives are met.')}")

        return "\n".join(lines)


def load_persona(filename: str) -> Optional[Persona]:
    """Load a persona from a JSON file."""

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

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    use_case = data.get('UseCase', {})
    assistant = use_case.get('Assistant', {})
    company = use_case.get('Company', {})

    return Persona(
        name=assistant.get('Name', 'Unknown'),
        role=assistant.get('Role', ''),
        personality=assistant.get('Personality', ''),
        voice_notes=assistant.get('VoiceNotes', ''),
        goal=assistant.get('Goal', ''),
        restrictions=assistant.get('Restrictions', ''),
        stages=data.get('Stages', []),
        company_name=company.get('CompanyName', ''),
        raw_data=data
    )


def list_personas() -> Dict[str, List[str]]:
    """List all available persona files, grouped by type."""
    if not PERSONAS_DIR.exists():
        return {"alice": [], "bob": []}

    personas = {"alice": [], "bob": []}

    for file in PERSONAS_DIR.glob("*.json"):
        name = file.stem
        if name.startswith("alice_"):
            personas["alice"].append(name)
        elif name.startswith("bob_"):
            personas["bob"].append(name)

    return personas


def get_default_personas() -> tuple:
    """Get the default persona filenames."""
    return ("alice_insurance_agent.json", "bob_frustrated_customer.json")
