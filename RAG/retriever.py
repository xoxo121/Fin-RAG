import os
import faiss
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_cohere import CohereRerank

from RAG.hyde import HyDEGenerator

load_dotenv()

INDEX_DIR = r"C:\Users\rahul\OneDrive\Documents\Artificial_Intelligence\LLM\Sarvam\Financial_RAG\RAG\indexes"
INDEX_NAME = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

class Retriever:
    def __init__(
        self,
        hyde_model_name="llama3-8b-8192",
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        cohere_rerank_model="rerank-english-v3.0"
    ):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Load FAISS index
        self.vector_store = FAISS.load_local(
            os.path.join(INDEX_DIR, INDEX_NAME),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        # Retrieve doc_ids and get corresponding Documents
        all_ids = list(self.vector_store.index_to_docstore_id.values())
        all_docs = []
        for doc_id in all_ids:
            doc = self.vector_store.docstore.search(doc_id)
            if isinstance(doc, Document):
                all_docs.append(doc)

        # Initialize BM25 retriever with all documents
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = 20

        # Vector retriever from FAISS
        retriever_vectordb = self.vector_store.as_retriever(search_kwargs={"k": 20})

        # Hybrid (ensemble) retrieval: FAISS + BM25
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vectordb, self.bm25_retriever],
            weights=[0.5, 0.5]
        )

        # HyDE generator
        self.hyde = HyDEGenerator(model_name=hyde_model_name)
        # reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        # self.reranker_compressor = CrossEncoderReranker(model=reranker, top_n=10)
        if cohere_api_key:
            os.environ["COHERE_API_KEY"] = cohere_api_key

        # Initialize Cohere reranker
        self.compressor = CohereRerank(model=cohere_rerank_model, top_n=12)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.ensemble_retriever
        )

    def retrieve(self, query: str, top_k: int = 10, filter_metadata: dict = {}):
        """
        Perform a HyDE-guided hybrid retrieval (FAISS + BM25), then re-rank with Rerank.

        Args:
            query (str): The user's query.
            top_k (int): Number of top documents to return.
            filter_metadata (dict, optional): A dictionary of metadata filters. Only documents
                                              whose metadata match all these filters are returned.

        Returns:
            List[Document]: The top-k documents after re-ranking.
        """
        # Generate a hypothetical doc (HyDE)
        hypothetical = self.hyde.generate_hypothetical_document(query)
        print(hypothetical)
        # Retrieve and rerank
        docs = self.compression_retriever.invoke(hypothetical)

        # Filter by metadata if requested
        if filter_metadata:
            filtered_docs = []
            for d in docs:
                if all(d.metadata.get(k) == v for k, v in filter_metadata.items()):
                    filtered_docs.append(d)
            docs = filtered_docs
        print(docs)

        return docs[:top_k]
