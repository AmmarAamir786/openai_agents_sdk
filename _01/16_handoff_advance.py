import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, RunContextWrapper, Runner, function_tool, handoff

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

class ManagerEscalation(BaseModel):
    issue: str # the issue being escalated
    why: str # why can you not handle it? Used for training in the future

@function_tool
def create_ticket(issue: str):
    """"
    Create a ticket in the system for an issue to be resolved.
    """
    print(f"Creating ticket for issue: {issue}")
    return "Ticket created. ID: 12345"
    # In a real-world scenario, this would interact with a ticketing system

manager_agent = Agent(
    name="Manager",
    handoff_description="Handles escalated issues that require managerial attention",
    instructions=(
        "You handle escalated customer issues that the initial custom service agent could not resolve. "
        "You will receive the issue and the reason for escalation. If the issue cannot be immediately resolved for the "
        "customer, create a ticket in the system and inform the customer."
    ),
    tools=[create_ticket],
)

def on_manager_handoff(ctx: RunContextWrapper[None], input: ManagerEscalation):
    print("Escalating to manager agent: ", input.issue)
    print("Reason for escalation: ", input.why)
    # here we might store the escalation in a database or log it for future reference


customer_service_agent = Agent(
    name="Customer Service",
    instructions="You assist customers with general inquiries and basic troubleshooting. " +
                 "If the issue cannot be resolved, escalate it to the Manager along with the reason why you cannot fix the issue yourself.",
    handoffs=[handoff(
        agent=manager_agent,
        input_type=ManagerEscalation,
        on_handoff=on_manager_handoff,
    )]
)

async def main():
    
    result = await Runner.run(
        customer_service_agent, 
        "I want a refund, but your system wont let me process it. Let me talk to the manager",
        run_config=run_config,
    )
    
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
