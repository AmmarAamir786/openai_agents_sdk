import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem
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

# 4 Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant."
)

async def main():
    convo: list[TResponseInputItem] = []

    print("You are now chatting with an assistant agent. Type 'exit' to end the conversation.")

    while True:
        user_input = input("Your Message: ")
        print("User: " + user_input)

        if user_input == "exit":
            print("Goodbye!")
            break

        print(convo)

        convo.append({"content": user_input, "role": "user"})
        print(convo)
        
        result = await Runner.run(
            agent,
            input=convo,
            run_config=run_config,
        )

        print(f"Agent Response: {result.final_output}")

        convo = result.to_input_list() # Overide the whole convo history

        print(convo)


if __name__ == "__main__":
    asyncio.run(main())