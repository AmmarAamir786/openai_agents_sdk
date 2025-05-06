import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner

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

class Story(BaseModel):
    title: str
    outline: str
    story: str

# Agent 3: Write a short story based on the validated outline.
story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the following outline: {input}",
    handoff_description="Writes story based on the outline provided.",
    output_type=Story,
)


# Agent 1: Generate a story outline from a given prompt.
story_outline_agent = Agent(
    name="story_outline_agent",
    instructions=(
        "Generate a brief and creative story outline based on the following prompt: {input}"
        "After generating the outline, hand it off to the story_agent."
    ),
    handoffs=[story_agent]
)

async def main():
    # Prompt user for the story idea
    prompt = input("Enter a prompt for the story: ")

    # Step 1: Generate the story outline.
    story = await Runner.run(
        story_outline_agent,
        prompt,
        run_config=run_config,
    )
    generated_story = story.final_output
    print("\nGenerated story:")
    print(generated_story)


if __name__ == "__main__":
    asyncio.run(main())
