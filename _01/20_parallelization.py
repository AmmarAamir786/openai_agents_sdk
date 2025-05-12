import asyncio
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
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

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Spanish translation from the given options.",
)

async def main():
    msg = input("Hi! Enter a message, and we'll translate it to Spanish.\n\n")

    # Ensure the entire workflow is a single trace
    res_1, res_2, res_3 = await asyncio.gather(
        Runner.run(
            spanish_agent,
            msg,
            run_config=run_config,
        ),
        Runner.run(
            spanish_agent,
            msg,
            run_config=run_config,
        ),
        Runner.run(
            spanish_agent,
            msg,
            run_config=run_config,
        ),
    )

    outputs = [
        ItemHelpers.text_message_outputs(res_1.new_items),
        ItemHelpers.text_message_outputs(res_2.new_items),
        ItemHelpers.text_message_outputs(res_3.new_items),
    ]

    translations = "\n\n".join(outputs)
    print(f"\n\nTranslations:\n\n{translations}")

    best_translation = await Runner.run(
        translation_picker,
        f"Input: {msg}\n\nTranslations:\n{translations}",
        run_config=run_config,
    )

    print(f"Best translation: {best_translation.final_output}")

if __name__ == "__main__":
    asyncio.run(main())