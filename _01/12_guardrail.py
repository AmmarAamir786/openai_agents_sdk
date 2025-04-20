import asyncio
from agents import Agent, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, InputGuardrail, GuardrailFunctionOutput, TResponseInputItem
from agents.run import RunConfig
from pydantic import BaseModel
from dotenv import load_dotenv
import os

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

# Translation agents for Spanish, French, and Roman Urdu
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

# Define a Pydantic output model for translation guardrail
class TranslationOutput(BaseModel):
    is_translation: bool
    reasoning: str

# Create an agent to act as our guardrail for translation requests
translation_guardrail_agent = Agent(
    name="Translation Guardrail",
    instructions=(
        "Check if the user's message is a translation request. "
        "Return a JSON object with 'is_translation': true if it is a translation request, "
        "or false otherwise, and provide your reasoning in the 'reasoning' field."
    ),
    output_type=TranslationOutput,
    model=model
)

# Define the guardrail function using the guardrail agent
async def translation_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input_data: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(
        translation_guardrail_agent, 
        input_data, 
        context=ctx.context)
    
    final_output = result.final_output_as(TranslationOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_translation,
    )

# Update the triage agent to include the input guardrail.
# Before routing the query to one of the translation agents, the guardrail checks whether
# the input is indeed a translation request.
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's translation request",
    handoffs=[spanish_agent, french_agent, urdu_agent],
    input_guardrails=[InputGuardrail(guardrail_function=translation_guardrail)]
)

async def main():
    result = await Runner.run(
        triage_agent,
        input="who is the president of pakistan",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())