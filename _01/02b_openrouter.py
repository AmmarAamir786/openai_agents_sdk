import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

# Load the environment variables from the .env file
load_dotenv()

# Set up the your Gemini API key
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

# 1 Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)
    
# 2 Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='meta-llama/llama-4-maverick:free',
    openai_client=provider,
)

# 3 Set up the run configuraion
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)
  
# 4 Set up the agent to use the model
query_agent = Agent(
    name="query_agent",
    instructions="You are a helpful assistant.",
)

async def main():
    # 5 Set up the runner to use the agent
    result = await Runner.run(
        query_agent,
        input="hi",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())