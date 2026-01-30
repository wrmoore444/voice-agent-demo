"""Analyzes conversation pace/energy to determine audio overlap timing."""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PaceAnalysis:
    """Result of pace analysis."""
    pace: float  # 0.0 (slow/calm) to 1.0 (fast/excited)
    energy: str  # "calm", "normal", "energetic", "heated"
    reason: str  # Brief explanation


# Words/phrases that indicate high energy
HIGH_ENERGY_WORDS = {
    # Excitement
    "wow", "whoa", "amazing", "incredible", "awesome", "fantastic", "crazy",
    "insane", "wild", "unbelievable", "seriously", "literally", "totally",
    # Urgency/intensity
    "wait", "hold on", "oh my", "oh no", "what", "really", "actually",
    "exactly", "yes", "no way", "come on", "get this",
    # Agreement/disagreement intensity
    "absolutely", "definitely", "completely", "exactly", "totally",
    "disagree", "wrong", "right",
    # Emotional
    "love", "hate", "can't believe", "so cool", "so weird",
}

# Words/phrases that indicate calm/thoughtful pace
CALM_WORDS = {
    "well", "hmm", "interesting", "i think", "perhaps", "maybe",
    "consider", "honestly", "actually", "in a way", "i suppose",
    "fair point", "true", "i see", "that makes sense",
}


def analyze_pace(text: str, previous_texts: List[str] = None) -> PaceAnalysis:
    """
    Analyze the pace/energy of a text response.

    Returns a PaceAnalysis with:
    - pace: 0.0 (very calm) to 1.0 (very excited/heated)
    - energy: categorical label
    - reason: why this pace was chosen
    """
    text_lower = text.lower()
    words = text_lower.split()

    # Initialize score at neutral
    score = 0.5
    reasons = []

    # Factor 1: Exclamation marks (big energy indicator)
    exclamations = text.count('!')
    if exclamations >= 2:
        score += 0.25
        reasons.append(f"{exclamations} exclamation marks")
    elif exclamations == 1:
        score += 0.1
        reasons.append("exclamation mark")

    # Factor 2: Question marks (engagement/back-and-forth)
    questions = text.count('?')
    if questions >= 1:
        score += 0.1
        reasons.append("question")

    # Factor 3: Sentence length (shorter = faster pace)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_words <= 5:
            score += 0.15
            reasons.append("very short sentences")
        elif avg_words <= 8:
            score += 0.08
            reasons.append("short sentences")
        elif avg_words >= 15:
            score -= 0.1
            reasons.append("long sentences")

    # Factor 4: High energy words
    high_energy_count = sum(1 for word in HIGH_ENERGY_WORDS if word in text_lower)
    if high_energy_count >= 3:
        score += 0.2
        reasons.append(f"{high_energy_count} high-energy words")
    elif high_energy_count >= 1:
        score += 0.1
        reasons.append("high-energy word")

    # Factor 5: Calm/thoughtful words
    calm_count = sum(1 for word in CALM_WORDS if word in text_lower)
    if calm_count >= 2:
        score -= 0.15
        reasons.append(f"{calm_count} thoughtful words")
    elif calm_count >= 1:
        score -= 0.08
        reasons.append("thoughtful word")

    # Factor 6: ALL CAPS words (shouting/emphasis)
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    if caps_words >= 1:
        score += 0.15
        reasons.append("emphasized words")

    # Factor 7: Ellipsis (trailing off = slower)
    if '...' in text:
        score -= 0.1
        reasons.append("ellipsis")

    # Factor 8: Text length (very short responses = quick back-and-forth)
    if len(text) < 30:
        score += 0.1
        reasons.append("very brief")
    elif len(text) < 50:
        score += 0.05
        reasons.append("brief")

    # Factor 9: Interruption patterns
    interruption_starters = ["but ", "wait ", "hold on", "no ", "yes ", "oh "]
    if any(text_lower.startswith(s) for s in interruption_starters):
        score += 0.1
        reasons.append("interruption pattern")

    # Clamp to 0.0-1.0
    score = max(0.0, min(1.0, score))

    # Determine energy category
    if score >= 0.75:
        energy = "heated"
    elif score >= 0.55:
        energy = "energetic"
    elif score >= 0.35:
        energy = "normal"
    else:
        energy = "calm"

    reason = ", ".join(reasons) if reasons else "neutral tone"

    return PaceAnalysis(pace=score, energy=energy, reason=reason)


def pace_to_overlap_ms(pace: float) -> int:
    """
    Convert pace (0.0-1.0) to overlap in milliseconds.

    - pace 0.0 (calm) → -100ms (slight overlap)
    - pace 0.5 (normal) → -300ms (moderate overlap)
    - pace 1.0 (heated) → -600ms (jumping in, talking over each other)
    """
    if pace <= 0.5:
        # Calm to normal: -100ms to -300ms
        return int(-100 - (pace * 400))
    else:
        # Normal to heated: -300ms to -600ms
        return int(-300 - ((pace - 0.5) * 600))


def pace_to_overlap_ms_with_variance(pace: float) -> Tuple[int, int]:
    """
    Get overlap range (min, max) based on pace for natural variation.
    """
    base = pace_to_overlap_ms(pace)
    variance = 150  # +/- 150ms variance
    return (base - variance, base + variance)
