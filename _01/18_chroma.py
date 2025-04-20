import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
from agents.tool import function_tool
import asyncio

# ──────────────────────────────────────────────
# 1. Load environment variables
# ──────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# ──────────────────────────────────────────────
# 2. Gemini Embedding Function
# ──────────────────────────────────────────────
def get_gemini_embedding(text: str) -> list[float]:
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response['embedding']

# ──────────────────────────────────────────────
# 3. Load and Chunk PDF
# ──────────────────────────────────────────────
reader = PdfReader("Panaversity.pdf")
full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

# ──────────────────────────────────────────────
# 4. Create Chroma Collection and Embed Chunks
# ──────────────────────────────────────────────
client = chromadb.Client()
collection = client.get_or_create_collection(name="my_documents")

# Clear old data
all_ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
collection.delete(ids=all_ids)

# Embed and store chunks
for i, chunk in enumerate(chunks):
    embedding = get_gemini_embedding(chunk)
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"doc_chunk_{i}"]
    )

# ──────────────────────────────────────────────
# 5. Create Tool for Chroma Search
# ──────────────────────────────────────────────
@function_tool("chroma_search")
def chroma_search(query: str) -> str:
    """
    Search relevant content from the Panaversity document using ChromaDB.
    """
    query_embedding = get_gemini_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n\n".join(results['documents'][0])

# ──────────────────────────────────────────────
# 6. Setup Agent and Runner
# ──────────────────────────────────────────────
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

agent = Agent(
    name="Document QA Agent",
    instructions="You are a helpful assistant who answers questions based only on the provided document.",
    tools=[chroma_search],
)

# ──────────────────────────────────────────────
# 7. Run the Agent
# ──────────────────────────────────────────────
async def main():
    result = await Runner.run(
        agent,
        input="What does the document say about courses offered?",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())


# Without Persistence:
# Every time you run the script:

# It reads the PDF.

# Re-chunks it.

# Re-generates embeddings.

# Re-adds all chunks into ChromaDB.

# This is slow and wastes compute/API calls (especially if you’re using paid embeddings like Google Gemini).