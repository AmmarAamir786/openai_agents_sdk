uv add openai-agents
uv add dotenv
set env

uv add chainlit
uv run chainlit hello
create chatbot.py
uv run chainlit run chatbot.py -w