from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Set up the your Gemini API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
# groq_api_key = os.getenv('GROQ_API_KEY')

# 1 Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# provider = AsyncOpenAI(
#     api_key=groq_api_key,
#     base_url="https://api.groq.com/openai/v1",
# )

# 2 Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

# model = OpenAIChatCompletionsModel(
#     model='llama-3.2',
#     openai_client=provider,
# )

# 3 Set up the run configuraion
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# 4 Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a joker. you write jokes"
)

# 5 Set up the runner to use the agent
result = Runner.run_sync(
    agent,
    input="what is the meaning of life?",
    run_config=run_config,
)

print(result.final_output)