# Handoffs:
# In a handoff, the parent (or triage) agent forwards the entire conversation history to a child agent. 
# The child agent then “takes over” the conversation. 
# In your example, the triage agent examines the input and passes the conversation to one of the translation agents. 
# They work as independent responders, and the conversation context is seamlessly passed along.

# Agents as Tools:
# When you transform an agent into a tool using the as_tool method, you make it callable like a function. 
# In this setup, the original agent (or another agent) stays in charge of the conversation. 
# It simply “calls” the tool with a generated prompt. 
# The tool returns only its final output, and that result is then integrated into the parent conversation. 
# There is no complete transfer of conversational control.

import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the asynchronous provider and model
provider = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Define individual translation agents.
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An English to Spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An English to French translator",
    model=model
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An English to Italian translator",
    model=model
)

# Compose an orchestrator agent that uses the above agents as tools.
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools provided to translate messages. "
        "If multiple translations are requested, call the relevant tools in order. "
        "Never translate by yourself; always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
    model=model
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's query",
    handoffs=[spanish_agent, french_agent, italian_agent],
)

async def main():

    # handoff_result = await Runner.run(
    #     triage_agent,
    #     input="please translate this to spanish: 'Hello, how are you?'",
    #     run_config=run_config,
    # )
    # print("Handoff Result:", handoff_result.final_output)
    
    # Instead of calling the tool directly (which causes the error), invoke the orchestrator.
    orchestrator_result = await Runner.run(
        orchestrator_agent,
        input="Translate 'Good morning, have a nice day!' to Spanish",
        run_config=run_config,
    )
    print("Orchestrator Result:", orchestrator_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
