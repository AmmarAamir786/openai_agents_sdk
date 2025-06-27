import asyncio
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
from agents.tool import function_tool
import os
# from agents import enable_verbose_stdout_logging
# enable_verbose_stdout_logging()

# Load the environment variables from the .env file
load_dotenv()

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



async def main():
    # 4 Set up the agent to use the model
    agent = Agent(
        name="agent",
        instructions="You are a helpful assistant.",
        # tools=[tool_a],
        model_settings=ModelSettings(tool_choice="required"),
        tool_use_behavior="stop_on_first_tool",
    )

    # 5 Set up the runner to use the agent
    result = await Runner.run(
        agent, 
        input="hello", 
        run_config=run_config
    )

    print(result.final_output)

if __name__ == "__main__":
    print("Starting the async agent...")
    asyncio.run(main())
    print("Async agent finished.")