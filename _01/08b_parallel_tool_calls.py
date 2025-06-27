import asyncio
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
from agents.tool import function_tool
import os
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

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

# Even if model settings is no set, the agent will still use parallel tool calls

#creating tools
@function_tool("get_weather")
def get_weather(location: str, unit: str = "Celcius") -> str:
  """
  Fetch the weather for a given location, returning a short description.
  """
  # Example logic
  return f"The weather in {location} is 22 degrees {unit}."


@function_tool("piaic_student_finder")
def piaic_student_finder(student_roll: int) -> str:
  """
  find the PIAIC student based on the roll number
  """
  data = {1: "Qasim",
          2: "Sir Zia",
          3: "Daniyal"}

  return data.get(student_roll, "Not Found")


async def main():
    # 4 Set up the agent to use the model
    agent = Agent(
        name="agent",
        instructions="You are a helpful assistant.",
        tools=[get_weather, piaic_student_finder],
        # model_settings=ModelSettings(parallel_tool_calls=True)
    )

    # 5 Set up the runner to use the agent
    result = await Runner.run(
        agent, 
        input="what is the name of the student with roll number 2 and what is the weather in karachi", 
        run_config=run_config
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())