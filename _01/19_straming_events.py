import asyncio
import random
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
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

@function_tool
def how_many_jokes() -> int:
    return random.randint(1, 10)

# 4 Set up the agent to use the model
agent = Agent(
    name="Joker",
    instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
    tools=[how_many_jokes],
)

async def main():
    # 5 Set up the runner to use the agent
    result = Runner.run_streamed(
        agent,
        input="Tell jokes",
        run_config=run_config,
    )

    print("=== Run starting ===")
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())