import os
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from agents.tool import function_tool
from openai import AsyncOpenAI

# ──────────────────────────────────────────────
# 1. Load environment variables
# ──────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# ──────────────────────────────────────────────
# 2. Embedding function
# ──────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ──────────────────────────────────────────────
# 3. Vector store path and loading logic
# ──────────────────────────────────────────────
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "my_documents"

chroma_metadata_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")

if os.path.exists(chroma_metadata_file):
    print("🔁 Loading existing ChromaDB vectorstore...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
else:
    print("📄 No existing vectorstore found. Embedding and creating ChromaDB store...")
    reader = PdfReader("Panaversity.pdf")
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print("✅ ChromaDB vectorstore created and persisted.")

# ──────────────────────────────────────────────
# 4. Create a Retrieval Tool
# ──────────────────────────────────────────────
@function_tool("chroma_search")
def chroma_search(query: str) -> str:
    """
    Search relevant content from the Panaversity document using ChromaDB.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# ──────────────────────────────────────────────
# 5. Setup Agent and Runner
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
    instructions="""
                    You are a helpful assistant. 
                    You must answer user queries using only the information available in the document.
                    To do so, always use the `chroma_search` tool.
                    Do not rely on prior knowledge or assumptions. Just invoke the tool with the user query and use its results to answer.
                """,
    tools=[chroma_search],
)

# ──────────────────────────────────────────────
# 6. Run the Agent
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


# With Persistence:
# The ChromaDB vector store is saved to disk in a local folder (e.g., ./chroma_db).

# On future runs, if that folder exists:

# You can load the pre-existing vector store directly.

# No need to re-process the PDF or re-embed anything unless the document changes.

# Saves you:

# ⏱ Time

# 💰 Cost (especially with APIs)

# 🔁 Redundant computation