# Memory-enhanced AI Assistant with Scheduling

An AI assistant with persistent memory and scheduling capabilities, powered by OpenAI's GPT models and mem0.

## Features

- Contextual Memory: Recalls past interactions
- Schedule Management: Add and retrieve schedule items
- Date Awareness: Provides current and specific date information
- Persistent Storage: Uses mem0 for long-term data retention

## Prerequisites

- Python 3.7+
- OpenAI API key
- Chroma DB

## Quick Start

1. Clone and install:
   ```
   git clone https://github.com/yurisasc/memory-enhanced-ai-assistant.git
   cd memory-enhanced-ai-assistant
   pip install mem0ai langchain_openai langchain_core langgraph chromadb python-dotenv gradio
   ```

2. Set up environment:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run Chroma DB:
   ```
   chroma run
   ```

4. Start the assistant:
   ```
   python assistant.py
   ```

## Usage Example

```
User: "Add a meeting with John on 2024-08-25 at 14:00 for 60 minutes."
Assistant: [Confirms meeting addition]

User: "What's my schedule for August 25, 2024?"
Assistant: [Provides schedule including the meeting with John]

[User restarts the application]

User: "When is my next meeting with John?"
Assistant: [Recalls and informs about the meeting on August 25, 2024]
```

The AI remembers schedules across sessions, allowing for persistent schedule management.

## Key Components

- `mem0`: Stores interactions and schedules
- `ChatOpenAI`: Uses GPT-4-0125-preview model
- `langgraph`: Manages conversation flow
- `gradio`: Provides user interface

## Customization

Extend capabilities by adding new tools in the script. Update the `tools` list and `system_message_content` in `call_model` function.

## Troubleshooting

- Ensure all packages are installed
- Verify OpenAI API key in `.env`
- Confirm Chroma DB is running

For issues, please open a GitHub issue.
