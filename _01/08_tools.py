import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
from agents.tool import function_tool
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
        instructions="You are a helpful assistant. Please provide clear answer without extra info. You will provide weather in celsius",
        tools=[get_weather, piaic_student_finder],
    )

    # 5 Set up the runner to use the agent
    result = Runner.run_streamed(
        agent, 
        input="what is the wheather in karachi", 
        run_config=run_config
    )

    # 6 Stream the response token by token
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
            token = event.data.delta
            print(token)

    # print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())