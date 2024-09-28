from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from llama_get_emb_func import get_llama_cpp_embeddings

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

embeddings = get_llama_cpp_embeddings()

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