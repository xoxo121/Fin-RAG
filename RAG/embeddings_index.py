import json
from pathlib import Path
from uuid import uuid4
import faiss
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

BASE_PATH = Path(r"C:\Users\rahul\OneDrive\Documents\Artificial_Intelligence\LLM\Sarvam\Financial_RAG\Parser\processed_data")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_DIR = Path(r"C:\Users\rahul\OneDrive\Documents\Artificial_Intelligence\LLM\Sarvam\Financial_RAG\RAG\indexes")
INDEX_NAME = "faiss_index"

# Mapping from company codes to company names
COMPANY_NAME_MAPPING = {
    "TCS": "TCS",
    "INFY": "Infosys",
    "HCLTECH": "HCLTech",
    "BHARTIARTL": "Bharti Airtel",
    "RELIANCE": "Reliance"
}

def extract_company_name(filename: str) -> str:
    for code, name in COMPANY_NAME_MAPPING.items():
        if code in filename:
            return name
    return "Unknown"

def build_index():
    all_documents = []
    # Iterate over each company's annual report filings
    for company_dir in BASE_PATH.iterdir():
        if company_dir.is_dir():
            chunks_file = company_dir / "chunks.json"
            if not chunks_file.exists():
                continue

            with chunks_file.open("r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            # Extract company name from the filename
            company_name = extract_company_name(company_dir.name)

            # Convert chunks to documents
            for c in chunk_data:
                doc_type = c.get("type", "text")
                title = c.get("title", "")
                text = c.get("text", "")
                reference = c.get("reference", "")
                summary = text.strip() if text.strip() else None

                # For embedding, we need some textual content. If it's a table with no summary,
                # use title+reference as fallback content.
                page_content = summary if summary else f"{title} {reference}"

                metadata = {
                    "company_name": company_name,
                    "type": doc_type,
                    "title": title,
                    "reference": reference
                }

                all_documents.append(Document(page_content=page_content, metadata=metadata))

    # Build embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    sample_emb = embeddings.embed_query("hello world")
    dimension = len(sample_emb)

    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    uuids = [str(uuid4()) for _ in all_documents]
    vector_store.add_documents(documents=all_documents, ids=uuids)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR / INDEX_NAME))
    print("Global index built with all companies' documents and tables successfully.")

if __name__ == "__main__":
    build_index()