import os
import gradio as gr
from embed_folder.docx_splitter import split_doc_from_headers

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from embed_folder.llama_get_emb_func import get_llama_cpp_embeddings
from embed_folder.pdf_splitter import Doc_pdf_parser
from langchain_chroma import Chroma

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from config import Config as cfg
from uuid import uuid4
from operator import itemgetter
import pandas as pd
import json

from embed_folder.minio_loader import MinioLoader
import time

import torch
import gc

def flush_VRAM():

    print('Flushing VRAM...')
    gc.collect()
    torch.cuda.empty_cache()
    gr.Info('VRAM flushed')
    print('VRAM flushed')

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

    gr.Info('Embedding complete!')

    return 


def get_client():
    persistent_client = chromadb.PersistentClient(
        path=cfg.CHROMA_PATH,
        settings=Settings(allow_reset=True),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    return persistent_client

def get_collection():
    collection = get_client().get_or_create_collection(cfg.COLLECTION_NAME)
    return collection

def get_vector_store():
    client = get_client()
    get_collection()

    embeddings = get_llama_cpp_embeddings()



    vector_store_from_client = Chroma(
        client = client,
        collection_name=cfg.COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vector_store_from_client

    
    




def get_df():
    wanted_columns = ['ids', 'metadatas', 'documents']

    res = dict(zip(wanted_columns, itemgetter(*wanted_columns)(get_collection().get())))
    # print(res)

    df = pd.DataFrame(res)
    df['documents'] = df['documents'].apply(lambda x: f'{x[:100]}...')
    df['metadatas'] = df['metadatas'].apply(lambda x: json.dumps(x))

    return df


def reset_and_create_chroma_client():
    persistent_client = get_client()
    
    persistent_client.reset()

    get_collection()

    print('Chroma client created')
    gr.Info('Chroma client created')


def simple_load(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    print(all_splits)

    uuids = [str(uuid4()) for _ in range(len(all_splits))]

    get_vector_store().add_documents(documents=all_splits, ids=uuids)


def load_url(url):
    print('Loading url...')

    loader = WebBaseLoader(url)
    data = loader.load()

    simple_load(data)

def load_pdf(url):
    print('Loading pdf...')

    # loader = PyPDFLoader(url)
    # data = loader.load()

    # simple_load(data)

    parser = Doc_pdf_parser(cfg.IMAGE_FOLDER)

    docs, ids = parser.parse_pdf(
        url, 
        skip_pages = [1, 2, 3, 4, 5])
    
    get_vector_store().add_documents(documents = docs, ids = ids)

def load_docx(url):
    print('Loading docx...')

    docs, ids = split_doc_from_headers(url)

    get_vector_store().add_documents(documents = docs, ids = ids)
    



def add_document(url):
    print(f'Embedding {url}..')
    gr.Info(f'Embedding {url}..')

    if url[-4:] == '.pdf':
        load_pdf(url)
    elif url[-5:] == '.docx':
        load_docx(url)
    else:
        load_url(url)

    print(f"Embedding of {url} is complete\n")
    return


def get_context_text(question: str, k: int = 1, score_max_thresh: int = 1) -> str:

    if not cfg.LOAD_EMBEDDINGS:
        return None

    if k <= 0:
        return None

    results = get_vector_store().similarity_search_with_score(
        question, 
        k=k, 
        #filter={"source": "news"}
    )

    docs = [(res, score) for (res, score) in results if score <= score_max_thresh]

    # for res, score in results:
    #     print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

    return docs



def get_context_image(context_list):
    try:

        most_valuable_context = context_list[0]

        prefix = 'http://localhost:9000/bucket1/pictures/'

    
        image_data = most_valuable_context[0].metadata['image']
        
        return prefix + image_data

    except Exception as e:
        print("Error while obtaining context image data:")
        print(e)

        return None




def format_context_to_log(context_list):
    output_str = ''

    if not context_list:
        return 'Context not found.'

    for ind, context in enumerate(context_list):
        doc = context[0]
        score = context[1]

        metadata = doc.metadata
        page_content = doc.page_content

        try:
            image = doc.metadata['image']
        except:
            image = None
        
        output_str += f'№ {ind+1}\nSCORE: {score:.3f}\nMETADATA: {metadata}\nIMAGE: {image}\nPAGE_CONTENT:\n{page_content}\n{"-"*40}\n\n'

    print(output_str)

    return output_str

def format_context_to_input(context_list):
    output_str = ''

    for ind, context in enumerate(context_list):

        output_str += f'{context[0].page_content}\n<next context>\n'

    return output_str


def upload_images_to_minio():

    mloader = MinioLoader()
    mloader.add_folder(cfg.IMAGE_FOLDER)