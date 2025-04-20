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

# Agent 1: Generate a story outline from a given prompt.
story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a brief and creative story outline based on the following prompt: {input}"
)

# Define a pydantic model to structure the output from the checker.
class OutlineCheckerOutput(BaseModel):
    is_good: bool
    is_scifi: bool

# Agent 2: Check the quality of the outline and determine if it's a sci-fi story.
outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions=(
        "Evaluate the following story outline for overall quality and whether it is a sci-fi outline. "
        "Return a JSON object with two boolean fields: 'is_good' and 'is_scifi'."
    ),
    output_type=OutlineCheckerOutput,
)

# Agent 3: Write a short story based on the validated outline.
story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the following outline: {input}",
    output_type=str,
)

async def main():
    # Prompt user for the story idea
    prompt = input("Enter a prompt for the story: ")

    # Step 1: Generate the story outline.
    outline_result = await Runner.run(
        story_outline_agent,
        prompt,
        run_config=run_config,
    )
    generated_outline = outline_result.final_output
    print("\nGenerated Outline:")
    print(generated_outline)

    # Step 2: Check the outline.
    checker_result = await Runner.run(
        outline_checker_agent,
        generated_outline,
        run_config=run_config,
    )
    checker_output = checker_result.final_output
    print("\nOutline Checker Output:")
    print(checker_output.model_dump_json())

    # # Step 3: Verify conditions for proceeding.
    if not checker_output.is_good:
        print("\nThe outline quality is not sufficient. Process stopped.")
        return
    if not checker_output.is_scifi:
        print("\nThe outline is not recognized as a sci-fi story. Process stopped.")
        return

    print("\nOutline validated as good quality and sci-fi. Proceeding to story generation.")

    # # Step 4: Generate the story based on the outline.
    story_result = await Runner.run(
        story_agent,
        generated_outline,
        run_config=run_config,
    )
    story = story_result.final_output
    print("\nGenerated Story:")
    print(story)

if __name__ == "__main__":
    asyncio.run(main())
