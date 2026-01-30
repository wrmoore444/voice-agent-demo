# Persona Testing Plan

## Overview
Test the ability to add new Stages files for each character and verify interactions work correctly across different persona combinations.

## Industry Structure

Each industry will have:
- **One agent persona** (customer service representative)
- **Multiple customer personas** (varying attitudes/situations)

### Example: Banking Industry
- Agent: `alice_bank_teller.json`
- Customers:
  - `bob_confused_customer.json` - doesn't understand a charge
  - `bob_impatient_customer.json` - in a hurry, needs quick resolution
  - (future) `bob_satisfied_customer.json` - routine inquiry, happy with service
  - (future) `bob_new_customer.json` - opening first account, lots of questions
  - (future) `bob_disgruntled_customer.json` - long-time customer with ongoing issues

### Example: Travel Industry
- Agent: `alice_travel_agent.json`
- Customers:
  - `bob_indecisive_traveler.json` - overwhelmed by choices
  - (future) `bob_budget_traveler.json` - price-conscious, wants deals
  - (future) `bob_luxury_traveler.json` - wants premium experience

### Example: Insurance Industry
- Agent: `alice_insurance_agent.json`
- Customers:
  - `bob_frustrated_customer.json` - claim denied, upset
  - (future) `bob_first_claim_customer.json` - never filed before, needs guidance

## Testing Checklist

### Stages File Creation
- [ ] Verify new JSON files load correctly via `load_persona()`
- [ ] Check system prompt generation includes all Stages
- [ ] Confirm DataPoints, Objectives, and ExamplePhrases are incorporated
- [ ] Test CompletionCriteria transitions between stages

### Interaction Testing
- [ ] Test each agent with each of their industry's customers
- [ ] Verify conversations stay on topic per Stages framework
- [ ] Check that different customer personalities produce distinct conversations
- [ ] Confirm agents adapt responses to customer tone/urgency

### Conversation Flow
- [ ] Agents should follow their Stage progression naturally
- [ ] Customers should exhibit their defined personality throughout
- [ ] Conversations should reach natural conclusions
- [ ] Stage transitions should feel organic, not forced

## UI Modifications Needed

### Current State
- `/demo/personas` returns flat list of all personas
- `/demo/start` accepts `alice` and `bob` parameters by filename

### Proposed Changes
1. **Group personas by industry**
   - Return personas organized: `{ "banking": { "agents": [...], "customers": [...] }, ... }`
   - Or add `industry` field to persona JSON files

2. **Industry selector in UI**
   - Dropdown to select industry first
   - Then show available agent (usually one) and customer options

3. **Persona preview**
   - Show persona summary before starting conversation
   - Display: name, role, personality, key objectives

4. **Conversation scenario labels**
   - Auto-generate labels like "Banking: Confused Customer"
   - Help users understand what scenario they're watching

## File Naming Convention

Proposed format: `{role}_{industry}_{personality}.json`

Examples:
- `agent_banking_teller.json`
- `customer_banking_confused.json`
- `customer_banking_impatient.json`
- `agent_travel_consultant.json`
- `customer_travel_indecisive.json`

Or keep current simple naming with metadata inside the JSON for filtering.

## Notes

- TTS is currently disabled for testing (API quota)
- Re-enable TTS before final interaction testing
- Consider recording sample conversations for each combo
