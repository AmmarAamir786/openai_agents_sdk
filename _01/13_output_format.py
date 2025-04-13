import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os

from pydantic import BaseModel

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

class WeatherAnswer(BaseModel):
  location: str
  temperature_c: float
  summary: str
  
# 4 Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant.",
    output_type=WeatherAnswer
)

async def main():
    # 5 Set up the runner to use the agent
    result = await Runner.run(
        agent,
        input="What's the temperature in Karachi?",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())