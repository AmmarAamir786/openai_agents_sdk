import os, asyncio, warnings, logging
from typing import List
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Quiet pdfplumber “CropBox missing” chatter
# ──────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="CropBox missing from /Page, defaulting to MediaBox",
    category=UserWarning,
)

# ──────────────────────────────────────────────
# PDF → text
# ──────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

# ──────────────────────────────────────────────
# Env & embeddings  (task_type removed)
# ──────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ──────────────────────────────────────────────
# Vector store (Chroma)
# ──────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME = "my_documents"
DB_FILE         = os.path.join(PERSIST_DIR, "chroma.sqlite3")

def build_documents(pdf_path: str) -> List[Document]:
    text      = extract_text_from_pdf(pdf_path)
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=1_000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks    = splitter.split_text(text)
    logging.info("🔹 Chunked into %s docs", len(chunks))
    return [Document(page_content=c) for c in chunks]

if os.path.exists(DB_FILE):
    print("🔁 Loading existing ChromaDB …")
    vectorstore = Chroma(
        collection_name    = COLLECTION_NAME,
        persist_directory  = PERSIST_DIR,
        embedding_function = embeddings,
    )
else:
    print("📄 Building ChromaDB …")
    docs        = build_documents("Panaversity.pdf")
    vectorstore = Chroma.from_documents(
        documents          = docs,
        embedding          = embeddings,
        collection_name    = COLLECTION_NAME,
        persist_directory  = PERSIST_DIR,
    )
    print("✅ ChromaDB stored.")

# ──────────────────────────────────────────────
# Retriever (MMR)
# ──────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "lambda_mult": 0.4, "fetch_k": 50},
)

def get_snippets(query: str, top_n: int = 5) -> List[str]:
    return [d.page_content.strip() for d in retriever.invoke(query)[:top_n]]

# ──────────────────────────────────────────────
# Tool for the agent
# ──────────────────────────────────────────────
from agents.tool import function_tool

@function_tool("chroma_search")
def chroma_search(query: str) -> str:
    """Return up to 5 relevant snippets from the Panaversity PDF."""
    snippets = get_snippets(query)
    return "NO_RELEVANT_CONTEXT" if not snippets else \
           "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

# ──────────────────────────────────────────────
# Agent & Runner
# ──────────────────────────────────────────────
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner

provider = AsyncOpenAI(
    api_key  = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)
model      = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)
run_config = RunConfig(model=model, model_provider=provider, tracing_disabled=True)

agent = Agent(
    name="Document QA Agent",
    instructions="""
Use ONLY information inside <context> tags and cite snippet numbers.
If <context> is NO_RELEVANT_CONTEXT, reply "Answer not found in the document."
<context>
{context}
</context>
""",
    tools=[chroma_search],
)

# ──────────────────────────────────────────────
# Main
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
