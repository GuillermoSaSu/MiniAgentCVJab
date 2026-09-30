# MiniAgentCVJab
Combines everything from the previous three projects into one bot: RAG over your CV, tool use, and conversation memory — Claude decides for itself which capability fits each question.

## Projects combined:

- [EasyClaudeChatbot](https://github.com/GuillermoSaSu/EasyClaudeChatbot)
- [ChatbotRAG](https://github.com/GuillermoSaSu/ChatbotRAG)
- [ChatBotToolUse](https://github.com/GuillermoSaSu/ChatBotToolUse)

## Getting it running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Drop your CV (PDF)** into `documents/`.

3. **Set `ANTHROPIC_API_KEY`.**

4. **Run:**
   ```bash
   python MiniAgent.py
   ```

5. Try mixing question types in one session:
   - "What experience do I have with Azure?" → `search_cv`
   - "What's the weather in Tokyo?" → `get_weather`
   - "What is a modular monolith?" → `search_wikipedia`
   - "What's 15% of 2400?" → `calculate`
   - "Given my Azure experience, is it currently raining in Madrid, and what time is it?" → **three tools in one question**

## The key design idea

RAG stops being "special" — it becomes just another tool (`search_cv`)
alongside the calculator, weather, time, and the new Wikipedia lookup.
Claude sees all 5 tool descriptions and picks whichever ones (if any) fit
the question, including calling several for one message.

## What's new here vs. the tool-use project

- **`search_cv` is a closure**: defined *inside* `main()`, after the CV has
  been chunked and embedded, so it can "see" `chunks` and `vectors` without
  needing to pass them around as extra parameters everywhere.
- **`search_wikipedia`**: a brand new tool hitting a real, free API
  (Wikipedia's REST summary endpoint, no key required) — gives the bot
  general knowledge lookup beyond both your CV and Claude's own training data.
