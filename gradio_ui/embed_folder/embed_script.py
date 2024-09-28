import os
import gradio as gr
from embed_folder.docx_splitter import split_doc_from_headers

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from embed_folder.llama_get_emb_func import get_llama_cpp_embeddings
from langchain_chroma import Chroma

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from config import Config as cfg
from uuid import uuid4
from operator import itemgetter
import pandas as pd


def update_output(files):
    if not files:
        return "No files uploaded"
    
    outputs = []

    for file in files:
        docs_list, _ = split_doc_from_headers(file)

        outputs.append("\n".join([str(doc) for doc in docs_list]))

    return "\n\n".join(outputs)

def add_documents(files):
    if not files:
        return "No files uploaded"

    for file in files:
        add_document(file)


    # for x in range(len(vector_store_from_client.get())):
    #     print(x)

    return 



persistent_client = chromadb.PersistentClient(
    path=cfg.CHROMA_PATH,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
collection = persistent_client.get_or_create_collection("docs")

embeddings = get_llama_cpp_embeddings()

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="docs",
    embedding_function=embeddings,
)

def get_df():
    wanted_columns = ['ids', 'metadatas', 'documents']

    res = dict(zip(wanted_columns, itemgetter(*wanted_columns)(collection.get())))
    print(res)

    df = pd.DataFrame(res)
    df['documents'] = df['documents'].apply(lambda x: f'{x[:100]}...')

    return df


def reset_and_create_chroma_client():
    global vector_store_from_client
    
    persistent_client.reset()

    persistent_client.get_or_create_collection('docs')

    vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name="docs",
        embedding_function=embeddings,
    )

    print('Chroma client created')


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


def get_context_text(question: str, k: int = 1, score_max_thresh: int = 1) -> str:

    if k <= 0:
        return None

    results = vector_store_from_client.similarity_search_with_score(
        question, 
        k=k, 
        #filter={"source": "news"}
    )

    docs = [(res, score) for (res, score) in results if score <= score_max_thresh]

    # for res, score in results:
    #     print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

    return docs


def format_context_to_log(context_list):
    output_str = ''

    if not context_list:
        return 'Context not found.'

    for ind, context in enumerate(context_list):
        doc = context[0]
        score = context[1]

        metadata = doc.metadata
        page_content = doc.page_content
        
        output_str += f'№ {ind+1}\nSCORE: {score:.3f}\nMETADATA: {metadata}\nPAGE_CONTENT:\n{page_content}\n{"-"*40}\n\n'

    print(output_str)

    return output_str

def format_context_to_input(context_list):
    output_str = ''

    for ind, context in enumerate(context_list):

        output_str += f'{context[0].page_content}\n<next context>\n'

    return output_str