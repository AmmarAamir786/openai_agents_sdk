import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    # Our custom logic goes here...
    await cl.Message(
        content=f"Received: {message.content}", # Send a fake response back to the user
    ).send()