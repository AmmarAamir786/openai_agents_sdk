import os, asyncio, warnings, logging
from typing import List
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Agent SDK
from agents.tool import function_tool
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()

# ── Silence noisy logs ────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message="CropBox missing from /Page", category=UserWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ── Config & keys ─────────────────────────────────────────────────────────
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

PDF_PATH        = Path(__file__).resolve().parent.parent / "Panaversity.pdf"
QDRANT_PATH     = "./qdrant_local"
COLLECTION_NAME = "panaversity_docs"

# ──────────────────────────────────────────────────────────────────────────
# 1.  Embeddings
# ──────────────────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ──────────────────────────────────────────────────────────────────────────
# 2.  Pre‑processing: Markdown‑aware chunking with heading metadata
# ──────────────────────────────────────────────────────────────────────────
def build_documents(pdf_path: Path) -> List[Document]:
    pages = PyPDFLoader(str(pdf_path)).load()
    md_text = "\n".join(p.page_content for p in pages)

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    big_chunks = md_splitter.split_text(md_text)

    fine_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    docs: List[Document] = []
    for bc in big_chunks:
        for chunk in fine_splitter.split_text(bc.page_content):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **bc.metadata,
                        "page": bc.metadata.get("page", "n/a"),
                        "source": "Panaversity.pdf",
                    },
                )
            )
    return docs

# ──────────────────────────────────────────────────────────────────────────
# 3.  Qdrant vector store (single client)
# ──────────────────────────────────────────────────────────────────────────
client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION_NAME):
    print("📄 Creating Qdrant collection …")
    dim = len(embeddings.embed_query("dimension_probe"))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    vectorstore.add_documents(build_documents(PDF_PATH))
    print("✅ Collection created and populated.")
else:
    print("🔁 Loaded existing Qdrant collection.")
    vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# ──────────────────────────────────────────────────────────────────────────
# 4.  Retriever: MMR only
# ──────────────────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 500, "lambda_mult": 0.3},
)

def get_snippets(query: str, top_n: int = 6) -> List[Document]:
    return retriever.invoke(query)[:top_n]

# ──────────────────────────────────────────────────────────────────────────
# 5.  Tool exposed to the agent
# ──────────────────────────────────────────────────────────────────────────
@function_tool("qdrant_search")
def qdrant_search(query: str) -> str:
    """Return up to 6 relevant snippets for the query."""
    docs = get_snippets(query)
    if not docs:
        return "NO_MATCH"
    return "\n\n".join(
        f"[{i+1} | p.{d.metadata.get('page')}] {d.page_content.strip()}"
        for i, d in enumerate(docs)
    )

# ──────────────────────────────────────────────────────────────────────────
# 6.  Agent setup (single Gemini call per request)
# ──────────────────────────────────────────────────────────────────────────
provider   = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
chat_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

run_cfg = RunConfig(model=chat_model, model_provider=provider, tracing_disabled=True)

agent = Agent(
    name="Panaversity Advisor",
    instructions="""
You are an academic counsellor for Panaversity.

**RULES**
1. ALWAYS call qdrant_search first.
2. If <context> is "NO_MATCH", reply:
   "I’m sorry – I couldn’t find that in the Panaversity document."
3. Otherwise answer using ONLY information inside <context>.
   • Cite snippet numbers like [2].  
   • If listing courses, format as a markdown table:

| Code | Title | Level | Duration/Credits |

<context>
{context}
</context>
""",
    tools=[qdrant_search],
)

# ──────────────────────────────────────────────────────────────────────────
# 7.  Demo run
# ──────────────────────────────────────────────────────────────────────────
async def main():
    res = await Runner.run(
        agent,
        input="hello",
        run_config=run_cfg,
    )
    print("\n=== Answer ===\n", res.final_output)

if __name__ == "__main__":
    asyncio.run(main())
