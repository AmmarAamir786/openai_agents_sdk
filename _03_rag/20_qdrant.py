import os, asyncio, warnings, logging
from typing import List
import pdfplumber
from dotenv import load_dotenv

# ── LangChain & embeddings ────────────────────────────────────────────────
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ── Agent SDK ─────────────────────────────────────────────────────────────
from agents.tool import function_tool
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

# ── Quiet noisy logs ──────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore", message="CropBox missing from /Page", category=UserWarning
)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ── Env & embeddings ──────────────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ── Helpers ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def build_documents(pdf_path: str) -> List[Document]:
    text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    return [Document(page_content=t) for t in splitter.split_text(text)]

# ── Qdrant local store ────────────────────────────────────────────────────
QDRANT_PATH, COLLECTION = "./qdrant_local", "panaversity_docs"
client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION):
    print("📄 Creating Qdrant collection …")
    dim = len(embeddings.embed_query("dimension_probe"))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )
    docs = build_documents("Panaversity.pdf")
    vectorstore.add_documents(docs)
    print("✅ Collection created and populated.")
else:
    print("🔁 Loaded existing Qdrant collection.")
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )

# ── Retriever (MMR) ───────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "lambda_mult": 0.4, "fetch_k": 50},
)
def get_snippets(query: str, top_n: int = 5):
    return [d.page_content.strip() for d in retriever.invoke(query)[:top_n]]

# ── Expose search tool to agent ───────────────────────────────────────────
@function_tool("qdrant_search")
def qdrant_search(query: str) -> str:
    snippets = get_snippets(query)
    return "NO_RELEVANT_CONTEXT" if not snippets else \
           "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

# ── Agent setup ───────────────────────────────────────────────────────────
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=provider
)
run_cfg = RunConfig(model=model, model_provider=provider, tracing_disabled=True)

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

# ── Run the agent ─────────────────────────────────────────────────────────
async def main():
    res = await Runner.run(
        agent,
        input="List all the courses that are offered in the document.",
        run_config=run_cfg,
    )
    print("\n=== Answer ===\n", res.final_output)

if __name__ == "__main__":
    asyncio.run(main())