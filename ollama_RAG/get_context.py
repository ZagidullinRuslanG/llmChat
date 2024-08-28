from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

import os
load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = 'docs'

persistent_client = chromadb.PersistentClient(
    
    path=CHROMA_PATH,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
    
)
collection = persistent_client.get_collection("docs")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="docs",
    embedding_function=embeddings,
)

def get_context_text(question: str, k: int = 1) -> str:

    results = vector_store_from_client.similarity_search_with_score(
        question, 
        k=k, 
        #filter={"source": "news"}
    )
    # for res, score in results:
    #     print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

    return results