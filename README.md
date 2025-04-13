uv add openai-agents
uv add dotenv
# set env

uv add chainlit
uv run chainlit hello
# create chatbot.py file
uv run chainlit run chatbot.py -w

# run openai files
uv run _01/01_sync.py
# change the file name after _01/ to run the file you want to run




