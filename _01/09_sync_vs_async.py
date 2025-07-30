import asyncio
import threading
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
from agents.tool import function_tool
import os
import time
# from agents import enable_verbose_stdout_logging
# enable_verbose_stdout_logging()

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

##################################
# 1st
##################################

# def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     time.sleep(5)
#     print("tool_a: Done blocking.")

# if __name__ == "__main__":
#     print("Starting the async agent...")
#     tool_a()
#     print("Async agent finished.")
    
##################################
# 2nd
##################################

# async def tool_b():
#     print("tool_b: Blocking the main thread for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_b: Done blocking.")

# if __name__ == "__main__":
#     print("Starting the async agent...")
#     asyncio.run(tool_b())
#     print("Async agent finished.")

##################################
# 3rd
##################################

# async def tool_c():
#     print("tool_c: Starting, will block for 3 seconds...")
#     await asyncio.sleep(3)
#     print("tool_c: Done blocking.")

# async def tool_d():
#     print("tool_d: Starting, will block for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_d: Done blocking.")

# async def main():
#     print("Starting both tools concurrently...")
#     await asyncio.gather(tool_c(), tool_d())
#     print("Both tools finished.")

# if __name__ == "__main__":
#     asyncio.run(main())

##################################
# 4th
##################################

# def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes"
#     )

#     result = Runner.run_sync(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the async agent...")
#     run_agent()
#     print("Async agent finished.")


##################################
# 5th
##################################

# async def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes"
#     )

#     result = await Runner.run(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the async agent...")
#     asyncio.run(run_agent())
#     print("Async agent finished.")

##################################
# Scenario 1:
#     run_sync with sync tool
##################################

# @function_tool()
# def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     time.sleep(5)
#     print("tool_a: Done blocking.")

# def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )

#     result = Runner.run_sync(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the agent...")
#     run_agent()
#     print("Agent finished.")


##################################
# Scenario 2:
#     run_sync with async tool
##################################

# @function_tool()
# async def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_a: Done blocking.")

# def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )

#     result = Runner.run_sync(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the agent...")
#     run_agent()
#     print("Agent finished.")

##################################
# Scenario 3:
#     run with sync tool
##################################

# @function_tool()
# def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     time.sleep(5)
#     print("tool_a: Done blocking.")

# async def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )

#     result = await Runner.run(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the agent...")
#     asyncio.run(run_agent())
#     print("Agent finished.")

##################################
# Scenario 4:
#     run with async tool
##################################

# @function_tool()
# async def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_a: Done blocking.")

# async def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )

#     result = await Runner.run(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the agent...")
#     asyncio.run(run_agent())
#     print("Agent finished.")


##################################
# Helpers For Testing
##################################

import asyncio, time, threading
from datetime import datetime

def ts() -> str:
    """Short timestamp helper."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(msg: str) -> None:
    """Print with thread info and timestamp."""
    print(f"[{ts()}] ({threading.current_thread().name}) {msg}")

async def ticker(label: str, interval: float = 0.5):
    """Coroutine that prints a dot at `interval` seconds."""
    while True:
        print(label, end="", flush=True)
        await asyncio.sleep(interval)
        
##################################
# Test Scenario 1:
#     run_sync with sync tool
##################################

# @function_tool()
# def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     time.sleep(5)
#     print("tool_a: Done blocking.")

# def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )

#     t0 = time.perf_counter()
    
#     result = Runner.run_sync(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )
    
#     log(f"run_sync() returned  (elapsed {time.perf_counter()-t0:.2f}s)")

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting Scenario1 demo …")
#     run_agent()
#     print("[main] Finished.")
    
# output =
# Starting Scenario1 demo …
# tool_a: Blocking the main thread for 5 seconds...
# tool_a: Done blocking.
# [00:18:48.388] (MainThread) run_sync() returned  (elapsed 6.06s)
# None
# [main] Finished

##################################
# Test Scenario 2:
#     run_sync with async tool
##################################

# @function_tool()
# async def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_a: Done blocking.")

# def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )
    
#     t0 = time.perf_counter()

#     result = Runner.run_sync(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )
    
#     log(f"run_sync() returned  (elapsed {time.perf_counter()-t0:.2f}s)")

#     print(result.final_output)

# if __name__ == "__main__":
#     print("Starting the agent...")
#     run_agent()
#     print("Agent finished.")
    
# output = 
# Starting the agent...
# tool_a: Blocking the main thread for 5 seconds...
# tool_a: Done blocking.
# [00:25:26.615] (MainThread) run_sync() returned  (elapsed 6.85s)
# None
# Agent finished.

##################################
# Test Scenario 3:
#     run with sync tool
##################################

# @function_tool()
# def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     time.sleep(5)
#     print("tool_a: Done blocking.")

# async def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )
    
#     # start a ticker so we can see if it freezes
#     tick = asyncio.create_task(ticker("."))
#     t0 = time.perf_counter()

#     result = await Runner.run(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )
    
#     log(f"Runner.run() done (elapsed {time.perf_counter()-t0:.2f}s)")

#     print(result.final_output)
    
#     tick.cancel()

# if __name__ == "__main__":
#     print("Starting the agent...")
#     asyncio.run(run_agent())
#     print("Agent finished.")
    
# output = 
# Starting the agent...
# ..tool_a: Blocking the main thread for 5 seconds...
# tool_a: Done blocking.
# .[00:33:07.490] (MainThread) Runner.run() done (elapsed 5.86s)
# None
# Agent finished

##################################
# Test Scenario 4:
#     run with async tool
##################################

# @function_tool()
# async def tool_a():
#     print("tool_a: Blocking the main thread for 5 seconds...")
#     await asyncio.sleep(5)
#     print("tool_a: Done blocking.")

# async def run_agent():
#     agent = Agent(
#         name="agent",
#         instructions="You are a joker. you write jokes",
#         model_settings=ModelSettings(tool_choice="tool_a"),
#         tool_use_behavior="stop_on_first_tool",
#         tools=[tool_a],
#     )
    
#     # start a ticker so we can see if it freezes
#     tick = asyncio.create_task(ticker("."))
#     t0 = time.perf_counter()

#     result = await Runner.run(
#         agent,
#         input="hi",
#         run_config=run_config,
#     )
    
#     log(f"Runner.run() done (elapsed {time.perf_counter()-t0:.2f}s)")

#     print(result.final_output)
    
#     tick.cancel()

# if __name__ == "__main__":
#     print("Starting the agent...")
#     asyncio.run(run_agent())
#     print("Agent finished.")
    
# output = 
# Starting the agent...
# ........tool_a: Blocking the main thread for 5 seconds...
# ..........tool_a: Done blocking.
# [00:47:03.400] (MainThread) Runner.run() done (elapsed 8.96s)
# None
# Agent finished.