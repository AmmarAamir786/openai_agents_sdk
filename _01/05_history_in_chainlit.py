from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import chainlit as cl
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Set up the your Gemini API key
gemini_api_key = os.getenv('GEMINI_API_KEY')

# 1 Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2 Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

# 3 Set up the run configuraion
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# 4 Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant."
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am a helpful assistant.").send()

@cl.on_message
async def main(message: cl.Message):

    # Get the history from the user session
    history = cl.user_session.get("history")

    # Append the user message to the history
    history.append({
        "role": "user",
        "content": message.content
    })

    # 5 Set up the runner to use the agent
    result = await Runner.run(
        agent,
        input=history,
        run_config=run_config,
    )

    # Append the llm output message to the history
    history.append({
        "role": "assistant",
        "content": result.final_output
    })
    
    await cl.Message(content=result.final_output).send()