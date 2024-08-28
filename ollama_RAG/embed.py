from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv
from uuid import uuid4
import chromadb
from docx_splitter import split_doc_from_headers

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
collection = persistent_client.get_or_create_collection("docs")
# collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="docs",
    embedding_function=embeddings,
)


from uuid import uuid4


def simple_load(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    print(all_splits)

    uuids = [str(uuid4()) for _ in range(len(all_splits))]

    vector_store_from_client.add_documents(documents=all_splits, ids=uuids)


def load_url(url):
    print('Loading url...')

    loader = WebBaseLoader(url)
    data = loader.load()

    simple_load(data)

def load_pdf(url):
    print('Loading pdf...')

    loader = PyPDFLoader(url)
    data = loader.load()

    simple_load(data)

def load_docx(url):
    print('Loading docx...')

    docs, ids = split_doc_from_headers(url)

    vector_store_from_client.add_documents(documents = docs, ids = ids)



def add_document(url):
    print(f'Embedding {url}..')

    if url[-4:] == '.pdf':
        load_pdf(url)
    elif url[-5:] == '.docx':
        load_docx(url)
    else:
        load_url(url)

    print("Embedding complete\n")
    return



# add_document(r'C:\Work\Gazprom\LLM\llmChat\data\Нефть.pdf')
# add_document(r'C:\Work\Gazprom\LLM\llmChat\data\днд.pdf')

add_document(r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\cand.docx')
add_document(r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\avr.docx')
add_document(r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\event.docx')