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
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

    # Gather full text and track page numbers
    pages = [
        (i + 1, page.extract_text())
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]

    full_text = "\n".join([text for _, text in pages])
    raw_documents = [Document(page_content=full_text)]

    # Use LangChain's better text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    # Split into chunks
    split_chunks = text_splitter.split_documents(raw_documents)

    # Add metadata to each chunk (we’ll assign dummy page based on character offset)
    documents = []
    char_count = 0
    for i, chunk in enumerate(split_chunks):
        # Roughly assign page number by cumulative char position
        page = next((pg for pg, txt in pages if char_count < len(txt)), 1)
        doc = Document(
            page_content=f"Document: {chunk.page_content}",
            metadata={"page": page, "source": "Panaversity.pdf", "chunk_index": i}
        )
        documents.append(doc)
        char_count += len(chunk.page_content)

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
    instructions="You are a helpful assistant who answers questions based only on the provided document.",
    tools=[chroma_search],
)

# ──────────────────────────────────────────────
# 6. Run the Agent
# ──────────────────────────────────────────────
async def main():
    result = await Runner.run(
        agent,
        input="How many courses are being offered in panaversity?",
        run_config=run_config,
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())