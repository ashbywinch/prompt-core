# Product Specification: Conversational Evaluation Criteria Generator

## Overview
A tool that helps users create structured evaluation criteria for decision-making through natural conversation with an AI assistant.

## Problem Statement
When making important decisions (like choosing birthday presents, hiring candidates, or selecting products), people need clear criteria to evaluate options. Creating good criteria is hard - it requires thinking through what matters, prioritizing factors, and ensuring nothing important is missed.

## Solution
An AI-powered assistant that guides users through a conversation to define clear, structured evaluation criteria. The assistant asks questions, clarifies needs, and produces a well-organized criteria list ready for decision-making.

## User Experience

### Primary Use Case: Guided Criteria Creation
1. **User starts** with a general idea (e.g., "help me choose birthday presents for a 7-year-old")
2. **AI assistant asks questions** to understand requirements:
   - "What's your budget range?"
   - "Are there any age-appropriate considerations?"
   - "What interests or hobbies should we consider?"
3. **User responds naturally** in conversation
4. **Assistant continues asking** until it has enough information
5. **Assistant produces structured criteria** with:
   - Clear factor names (e.g., "budget", "age_appropriateness", "educational_value")
   - Descriptions of what each factor measures
   - Importance weights (1-10 scale)
   - Optional ideal/target values

### Key Features

#### 1. Natural Conversation Interface
- Chat-based interaction - users talk like they would with a human expert
- AI asks one question at a time to avoid overwhelming users
- Assistant understands context and builds on previous answers

#### 2. Structured Output Generation
- Always produces valid, complete criteria lists
- Ensures minimum requirements are met:
  - At least 2 evaluation factors
  - Always includes "budget" consideration (case-insensitive)
- Output is machine-readable for further analysis or integration

#### 3. Intelligent Limits
- **Max 10 conversation turns** - prevents endless loops
- **Three clear outcomes**:
  - ✅ **Success**: Valid criteria generated
  - 🔄 **Continue**: More questions needed
  - ❌ **Failure**: Can't proceed (user gives up or insufficient info)
- **Business rule enforcement**: Won't produce invalid criteria

#### 4. User-Friendly Error Handling
- Clear explanations when things go wrong
- No technical jargon in error messages
- Graceful failure with actionable suggestions

## Target Users

### Primary User: Decision Makers
- Individuals facing personal decisions (gifts, purchases, etc.)
- Professionals making business decisions (hiring, vendor selection)
- Teams needing collaborative evaluation frameworks

### Secondary User: Facilitators
- Coaches helping clients make decisions
- Consultants creating evaluation frameworks for clients
- Educators teaching decision-making skills

## Usage Scenarios

### Scenario 1: Personal Gift Selection
```
User: "I need help choosing a birthday gift for my 7-year-old niece"
Assistant: "I can help with that! What's your budget for this gift?"
User: "Around $50"
Assistant: "Great. Does she have any particular interests or hobbies?"
User: "She loves science and building things"
Assistant: "Perfect. Are there any safety considerations for her age?"
[Conversation continues...]
✅ Success: Generated criteria with factors: budget, educational_value, 
    safety, age_appropriateness, entertainment_value
```

### Scenario 2: Hiring Decision
```
User: "We're hiring a software developer"
Assistant: "What technical skills are most important for this role?"
User: "Python and cloud experience"
Assistant: "What about soft skills or team fit considerations?"
User: "Good communication and collaborative mindset"
[Conversation continues...]
✅ Success: Generated criteria with factors: technical_skills, 
    communication, collaboration, experience_level, cultural_fit
```

### Scenario 3: Product Comparison
```
User: "We need to choose a new project management tool"
Assistant: "What are your key requirements for this tool?"
User: "Must integrate with our existing systems"
Assistant: "What's your budget range for this tool?"
[Conversation continues...]
✅ Success: Generated criteria with factors: integration_capability, 
    budget, user_friendliness, feature_set, scalability
```

## Business Rules (Non-Negotiable)

### 1. Criteria Quality
- **Minimum 2 criteria**: Every decision needs at least 2 factors to compare
- **Must include "budget"**: Financial consideration is essential (case-insensitive)
- **Clear descriptions**: Each criterion explains what it measures
- **Weighted importance**: Each factor has 1-10 importance rating

### 2. Conversation Limits
- **Configurable limit to the number of conversation turns**: Prevents excessive API costs and user fatigue
- **Clear exit points**: Users can stop anytime, assistant can declare when stuck
- **No infinite loops**: Assistant recognizes unproductive conversations

### 3. Output Standards
- **Always valid structure**: Output always meets technical specifications
- **Machine-readable**: Can be exported, analyzed, or integrated
- **Human-friendly**: Clear, understandable format for users

## Success Metrics

### User Success Metrics
- **Completion rate**: Percentage of conversations that produce valid criteria
- **Turn efficiency**: Average turns needed to reach success
- **User satisfaction**: Qualitative feedback on helpfulness

### Technical Success Metrics  
- **API reliability**: Successful LLM interactions vs failures
- **Validation rate**: Percentage of generated criteria that pass business rules
- **Error clarity**: User understanding of error messages

## Integration Points

### Current
- **Command Line Interface**: Primary interaction method
- **Environment configuration**: API key management for different providers

### Future Potential
- The intention is to build a framework that allows the user and LLM to collaborate in making arbitrarily deep nested objects, for example, a business plan with many sections, where detailed validation criteria (both deterministic and otherwise) are applied to the structure and content.
- These objects may be saved as Markdown.
- The objects may consist of lists of proposed actions that can be reviewed and then be carried out programmatically (like a more structured version of tool calling)
- The functions that create these objects can be composed just like normal functions can, so our LLM object creating tool should be Turing complete.
- The system will be self modifying. After each session with the LLM, we'll work with the user to generate a list of system modifications that would make it produce better results on its next run.
- We will be using Temporal to provide checkpoint/restart functionality.

## Philosophy

### Design Principles
1. **Conversation First**: Natural dialogue over forms or checklists
2. **Structured Second**: Behind-the-scenes structure, front-end simplicity
3. **Fail Helpfully**: Clear explanations, not technical errors

### Technical Philosophy
- **Infrastructure exposure**: Tests fail when API keys missing (exposes setup issues)
- **Real validation**: Use actual LLMs to verify prompts work, not just mocks
- **Clean boundaries**: Clear separation of concerns. Low coupling. 

## Why This Matters

Good decisions require clear criteria. This tool makes criteria creation:
- **Accessible**: No expertise in evaluation frameworks needed
- **Thorough**: Systematic approach ensures nothing is missed  
- **Structured**: Results are usable for analysis and comparison
- **Adaptable**: Works for personal, professional, and team decisions

The conversation format makes what can feel like a dry, analytical task into an engaging, thoughtful dialogue that produces better outcomes.
