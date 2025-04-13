import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

provider = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
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
    instructions="You are a helpful assistant.",
)

async def main():
    # 5 Set up the runner to use the agent
    result = await Runner.run(
        agent,
        input="what is 2+2",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())