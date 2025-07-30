import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

provider = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    openai_client=provider,
)  # Automatically use tools when needed

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=model
)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model
)

#think of this as an orchestrator or supervisor agent
#It chooses which agent to send data to based on the user's query
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's query",
    handoffs=[spanish_agent, french_agent, urdu_agent],
)

async def main():
    result = await Runner.run(
        triage_agent,
        input="please translate this to urdu: 'Hello, how are you?'",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())