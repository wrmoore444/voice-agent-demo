# Persona Files

This folder contains persona JSON files that define bot personalities and conversation flows.

## File Naming Convention

- `alice_<role>.json` - Service agent personas (Alice initiates, provides service)
- `bob_<role>.json` - Customer personas (Bob responds, has a need/problem)
- `_TEMPLATE_new_persona.json` - Template for creating new personas

## Quick Start: Creating a New Persona

1. Copy `_TEMPLATE_new_persona.json`
2. Rename to `alice_<yourrole>.json` or `bob_<yourrole>.json`
3. Replace all `PLACEHOLDER` values with your content
4. Delete the `_README`, `_STAGE_COMMENT`, and `_TIPS` fields
5. Test with the demo viewer

## JSON Structure

```
{
  "UseCase": {
    "Company": { ... },      // Company/brand info
    "Assistant": { ... }     // Bot identity, personality, goals
  },
  "Stages": [ ... ]          // Conversation flow stages
}
```

### UseCase.Assistant Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `Name` | Bot's name | "Alice" |
| `Role` | Job title + context | "Senior Customer Service Rep at First National Bank" |
| `Personality` | How the bot comes across | "Trustworthy, efficient, knowledgeable" |
| `Goal` | What the bot tries to accomplish | "Help customers with banking needs efficiently" |
| `VoiceNotes` | Speaking style guidance | "Professional but warm tone. Use customer's name." |
| `Restrictions` | What the bot should NOT do | "Cannot approve loans on the spot" |

### Stage Fields

| Field | Purpose |
|-------|---------|
| `StageName` | Label for the stage |
| `Objectives` | What the bot should accomplish (list) |
| `DataPoints` | Information to collect (list of name/type/description) |
| `ExamplePhrases` | Sample utterances for inspiration (list) |
| `CompletionCriteria` | When to move to next stage |

### DataPoint Types

- `string` - Text (names, descriptions, etc.)
- `boolean` - True/false flags
- `number` - Numeric values
- `date` - Date values

### Using Variables in ExamplePhrases

Reference captured data points using `{DatapointName}`:

```json
"ExamplePhrases": [
  "I understand, {CustomerName}. Let me help you with that.",
  "Your appointment is confirmed for {AppointmentDate}."
]
```

## Alice vs Bob Personas

**Alice (Service Agent):**
- Stages represent the SERVICE FLOW (greeting → discovery → resolution → close)
- Objectives are about HELPING the customer
- DataPoints are information TO COLLECT

**Bob (Customer):**
- Stages represent the EMOTIONAL JOURNEY (frustrated → hopeful → satisfied)
- Objectives are about EXPRESSING needs and reactions
- DataPoints are information TO SHARE

## Existing Personas

### Alice Personas (Service Agents)
- `alice_bank_teller.json` - Bank customer service
- `alice_insurance_agent.json` - Insurance claims/policy support
- `alice_travel_agent.json` - Travel booking assistance

### Bob Personas (Customers)
- `bob_bank_upset_customer.json` - Frustrated about overdraft fees
- `bob_bank_confused_customer.json` - Confused about account features
- `bob_bank_new_customer.json` - New customer opening account
- `bob_insurance_frustrated_claimant.json` - Frustrated with claim process
- `bob_insurance_confused_policyholder.json` - Confused about coverage
- `bob_insurance_new_customer.json` - Shopping for insurance
- `bob_travel_angry_customer.json` - Flight was cancelled
- `bob_travel_first_time.json` - First-time traveler needs guidance
- `bob_travel_indecisive_planner.json` - Can't decide on destination

## Tips

1. **Keep it conversational** - Example phrases should sound natural, not scripted
2. **3-6 stages is typical** - Too few lacks depth, too many is overwhelming
3. **Test combinations** - Same Alice with different Bobs produces different conversations
4. **Iterate on prompts** - Watch conversations and refine based on what works
