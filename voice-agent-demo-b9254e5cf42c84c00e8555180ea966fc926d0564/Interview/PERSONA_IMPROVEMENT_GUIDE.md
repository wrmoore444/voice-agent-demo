# Persona Improvement Guide -ATEST

A systematic process for improving AI agent personas through automated testing and iterative refinement.

## Overview

This guide describes how to use the persona testing system to:
1. Run automated conversations between an agent and test customers
2. Analyze conversations against the agent's Stages file
3. Identify improvement opportunities
4. Refine the persona and re-test

This process works for any type of persona (customer service, sales, support, etc.) as long as you have:
- An **agent persona** (the AI you're improving)
- One or more **test personas** (simulated users/customers to interact with)

---

## Persona File Structure

Each persona is defined in a JSON file with two main sections:

### 1. Stages Array

Defines the conversation flow. Each stage contains:

```json
{
  "StageName": "Greeting",
  "DataPoints": [
    {
      "DatapointName": "CustomerName",
      "DatapointType": "string",
      "DatapointDescription": "What this data point captures"
    }
  ],
  "Objectives": [
    "What the agent should accomplish in this stage"
  ],
  "ExamplePhrases": [
    "Sample phrases the agent might use"
  ],
  "CompletionCriteria": "Conditions to move to the next stage"
}
```

### 2. UseCase Object

Defines the agent's identity and behavior:

```json
{
  "Company": {
    "CompanyName": "...",
    "ProductName": "...",
    "SuiteDescription": "..."
  },
  "Assistant": {
    "Name": "Agent name",
    "Role": "Agent's role/title",
    "Goal": "What the agent is trying to accomplish",
    "Personality": "How the agent behaves",
    "VoiceNotes": "Speaking style and tone guidance",
    "Restrictions": "What the agent cannot do"
  }
}
```

---

## Testing Process

### Step 1: Organize Your Personas

Place persona files in the `Interview/bot_demo/personas/` directory:

```
personas/
  alice_<industry>_<role>.json      # Agent personas
  bob_<industry>_<type>.json        # Test customer personas
```

The testing system automatically matches agents with test personas based on the industry tag in the filename.

**Examples:**
- `alice_travel_agent.json` matches with `bob_travel_*.json`
- `alice_insurance_agent.json` matches with `bob_insurance_*.json`

### Step 2: Run the Tests

Start the server and trigger a test run:

```bash
# Start the server
python Interview/main.py

# Run tests for a specific agent
curl -X POST "http://localhost:8001/persona-test/run/alice_travel_agent.json"
```

The system will:
1. Run a complete conversation with each matching test persona
2. Analyze each conversation against the agent's Stages file
3. Generate improvement suggestions
4. Save results to `Interview/bot_demo_pipecat/test_results/`

### Step 3: Review Results

Check available results:
```bash
curl "http://localhost:8001/persona-test/results"
```

Get a specific result:
```bash
curl "http://localhost:8001/persona-test/results/<filename>"
```

---

## Understanding the Analysis

Each test result contains:

### Per-Conversation Analysis

- **objectives_met**: Which stage objectives were achieved (with reasons)
- **data_points_captured**: Which data points were collected
- **issues**: Specific problems observed
- **suggestions**: Recommended improvements
- **overall_rating**: EXCELLENT, GOOD, NEEDS_IMPROVEMENT, or POOR

### Overall Analysis

- **COMMON_PATTERNS**: Issues that appeared across multiple conversations
- **PRIORITY_IMPROVEMENTS**: Ranked list of changes with impact assessment
- **STAGE_SPECIFIC_CHANGES**: Specific edits for each stage
- **OVERALL_ASSESSMENT**: Strengths and weaknesses summary

---

## Making Improvements

### Types of Improvements

#### 1. Add Missing Data Points

If the analysis shows the agent isn't capturing important information:

```json
{
  "DatapointName": "NewDataPoint",
  "DatapointType": "string",
  "DatapointDescription": "MANDATORY - Description of what to capture"
}
```

Use "MANDATORY" in the description to emphasize importance.

#### 2. Strengthen Objectives

Make objectives explicit and actionable:

**Before:**
```json
"Objectives": ["Help the customer"]
```

**After:**
```json
"Objectives": [
  "MANDATORY STEP 1: Identify the customer's specific need",
  "MANDATORY STEP 2: Verify identity before sharing details",
  "IF PROBLEM: Show empathy before offering solutions"
]
```

#### 3. Add Conditional Logic

Handle different scenarios:

```json
"Objectives": [
  "IF NEW CUSTOMER: Welcome and explain services",
  "IF EXISTING CUSTOMER: Verify identity and pull up account",
  "IF COMPLAINT: Acknowledge feelings before problem-solving"
]
```

#### 4. Improve Example Phrases

Add phrases that model the desired behavior:

```json
"ExamplePhrases": [
  "For your security, I need to verify your identity. Can you confirm...?",
  "Does that explanation make sense? I want to make sure you understand.",
  "Before we wrap up, let me summarize what we discussed..."
]
```

#### 5. Update Completion Criteria

Ensure stages can't be skipped:

```json
"CompletionCriteria": "CANNOT proceed until 'PolicyExplained' AND 'CustomerUnderstands' are true. THEN proceed to 'Close'."
```

#### 6. Strengthen VoiceNotes and Restrictions

Add explicit behavioral requirements:

```json
"VoiceNotes": "ALWAYS ask 'Does that make sense?' after explanations. ALWAYS summarize before ending. NEVER skip identity verification.",

"Restrictions": "You MUST collect email before confirming any booking. You MUST explain policies before closing."
```

---

## Iterative Improvement Cycle

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1. RUN TESTS                                          │
│      └─> Conversations with all test personas           │
│                                                         │
│   2. REVIEW ANALYSIS                                    │
│      └─> Check ratings, issues, suggestions             │
│                                                         │
│   3. IDENTIFY PRIORITIES                                │
│      └─> Focus on high-impact improvements first        │
│                                                         │
│   4. UPDATE PERSONA FILE                                │
│      └─> Add data points, strengthen objectives,        │
│          improve phrases, update criteria               │
│                                                         │
│   5. RE-TEST                                            │
│      └─> Verify improvements, check for regressions     │
│                                                         │
│   6. REPEAT until satisfied                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Common Improvement Patterns

### Pattern: Agent Skips Important Steps

**Symptom:** Analysis shows objectives not met or data points not captured.

**Fix:**
- Add "MANDATORY" to data point descriptions
- Add "MANDATORY STEP X:" to objectives
- Update completion criteria to block progression

### Pattern: Agent Doesn't Adapt to Situation

**Symptom:** Agent asks irrelevant questions or follows wrong path.

**Fix:**
- Add `CallType` or `SituationType` data point early
- Add conditional objectives: "IF X: do Y"
- Create separate paths in example phrases

### Pattern: Agent Ends Abruptly

**Symptom:** Conversation ends without proper wrap-up.

**Fix:**
- Add mandatory close steps (summary, email, additional help)
- Add `SummaryProvided`, `AdditionalHelpOffered` data points
- Update completion criteria to require these

### Pattern: Agent Doesn't Confirm Understanding

**Symptom:** Customer may not understand explanations.

**Fix:**
- Add "Ask 'Does that make sense?'" to objectives
- Add `UnderstandingConfirmed` data point
- Add example phrases that check comprehension

### Pattern: Agent Misses Emotional Cues

**Symptom:** Agent jumps to solutions without acknowledging feelings.

**Fix:**
- Add `EmotionalState` data point
- Add "Acknowledge feelings before solutions" to objectives
- Add empathetic example phrases

---

## Creating New Persona Types

To test a new type of agent:

1. **Create the agent persona** (`alice_<type>_<role>.json`)
   - Define stages appropriate to the use case
   - Set clear objectives and data points
   - Include realistic example phrases

2. **Create test personas** (`bob_<type>_<scenario>.json`)
   - Create 2-4 different user types (easy, difficult, confused, etc.)
   - Give them distinct behaviors and needs
   - Include realistic dialogue patterns

3. **Run initial tests** to establish a baseline

4. **Iterate** using the improvement cycle

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/persona-test/agents` | GET | List all testable agents with their matching customers |
| `/persona-test/run/{agent}` | POST | Run tests for an agent (runs in background) |
| `/persona-test/results` | GET | List all test result files |
| `/persona-test/results/{file}` | GET | Get a specific test result |

---

## Tips for Success

1. **Start simple** - Get basic conversation flow working before adding complexity
2. **Test frequently** - Small changes, frequent tests
3. **Focus on high-impact issues first** - Use the PRIORITY_IMPROVEMENTS ranking
4. **Use MANDATORY keywords** - They help the AI understand what's required
5. **Include explicit trigger phrases** - "Say 'Does that make sense?'" is clearer than "confirm understanding"
6. **Create diverse test personas** - Easy, difficult, confused, angry customers reveal different issues
7. **Check for regressions** - Improvements in one area shouldn't break another
