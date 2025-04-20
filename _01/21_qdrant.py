import os, asyncio, warnings, logging
from typing import List
from dotenv import load_dotenv
import pdfplumber

# LangChain & GenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

# Agent SDK
from agents.tool import function_tool
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

# ──────────────────────────────────────────────
# Silence pdfplumber CropBox warnings
# ──────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="CropBox missing from /Page, defaulting to MediaBox",
    category=UserWarning,
)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ──────────────────────────────────────────────
# Env & Embeddings
# ──────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ──────────────────────────────────────────────
# PDF → Text Extraction
# ──────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

# ──────────────────────────────────────────────
# Split into Documents
# ──────────────────────────────────────────────
def build_documents(pdf_path: str) -> List[Document]:
    text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    chunks = splitter.split_text(text)
    logging.info("🔹 Chunked into %s docs", len(chunks))
    return [Document(page_content=c) for c in chunks]

# ──────────────────────────────────────────────
# Qdrant (local embedded mode)
# ──────────────────────────────────────────────
QDRANT_PATH = "./qdrant_local"
COLLECTION_NAME = "panaversity_docs"

if os.path.exists(QDRANT_PATH):
    print("🔁 Loading Qdrant vectorstore …")
    vectorstore = Qdrant(
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        path=QDRANT_PATH,
    )
else:
    print("📄 Creating Qdrant vectorstore …")
    docs = build_documents("Panaversity.pdf")
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        path=QDRANT_PATH,
    )
    print("✅ Qdrant vectorstore created.")

# ──────────────────────────────────────────────
# Retriever with MMR
# ──────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "lambda_mult": 0.4, "fetch_k": 50},
)

def get_snippets(query: str, top_n: int = 5) -> List[str]:
    return [d.page_content.strip() for d in retriever.invoke(query)[:top_n]]

# ──────────────────────────────────────────────
# LangChain Tool exposed to agent
# ──────────────────────────────────────────────
@function_tool("qdrant_search")
def qdrant_search(query: str) -> str:
    """Return up to 5 relevant snippets from the Panaversity PDF."""
    snippets = get_snippets(query)
    return "NO_RELEVANT_CONTEXT" if not snippets else \
           "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

# ──────────────────────────────────────────────
# Agent Setup
# ──────────────────────────────────────────────
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
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
        Use ONLY information inside <context> tags and cite snippet numbers.
        If <context> is NO_RELEVANT_CONTEXT, reply "Answer not found in the document."
        <context>
        {context}
        </context>
    """,
    tools=[qdrant_search],
)

# ──────────────────────────────────────────────
# Run the Agent
# ──────────────────────────────────────────────
async def main():
    res = await Runner.run(
        agent,
        input="What does the document say about courses offered?",
        run_config=run_config,
    )
    print("\n=== Answer ===\n", res.final_output)

if __name__ == "__main__":
    asyncio.run(main())