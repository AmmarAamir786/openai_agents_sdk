import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner
)
from pydantic import BaseModel

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# One provider for both models
provider = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Define two Model implementations:
outline_model = OpenAIChatCompletionsModel(
    model="google/gemini-2.5-flash-preview",
    openai_client=provider,
)
story_model = OpenAIChatCompletionsModel(
    model="meta-llama/llama-4-maverick:free",
    openai_client=provider,
)

class Story(BaseModel):
    title: str
    outline: str
    story: str

# Agent 3: uses the Llama model to write the full story
story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the following outline: {input}",
    handoff_description="Writes story based on the outline provided.",
    output_type=Story,
    model=story_model,        # ← meta-llama here
)

# Agent 1: uses Gemini (which supports tool-calls) to draft the outline
story_outline_agent = Agent(
    name="story_outline_agent",
    instructions=(
    "Generate a brief and creative story outline based on the following prompt: {input}. "
    "Return the outline as a JSON object with fields: title, protagonist, antagonist, setting. "
    "After generating the outline, hand it off to the story_agent."
    ),
    handoffs=[story_agent],
    model=outline_model,      # ← Gemini here
)

async def main():
    prompt = "Time Travel"
    result = await Runner.run(
        story_outline_agent,
        prompt,
        run_config=RunConfig(tracing_disabled=True),
    )
    print("\nGenerated story:")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())