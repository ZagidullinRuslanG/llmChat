# requirements
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# Загрузка документов
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

# Эмбеддинги
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import LlamaCppEmbeddings

# QnA цепочка
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import torch
torch.cuda.is_available()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


model_path = r"C:/Work/Gazprom/LLM/llmChat/data/weights/openchat_3.5.Q5_K_M.gguf"

rag_template: str = (
            'Отвечай на поставленные вопросы.\n'
            '\n'
            '{question}'
        )

rag_prompt = ChatPromptTemplate.from_template(rag_template)

callbacks = [StreamingStdOutCallbackHandler()]

model = LlamaCpp(
    model_path=model_path, 
    temperature=0.5,
    # max_new_tokens=1000,
    # context_window=16379-1000,
    generate_kwargs={},
    # n_ctx=8192,
    n_gpu_layers=-1, 
    # n_threads=6, 
    n_batch=512, 
    verbose=True,
    callback_manager=callback_manager
)


chain = (
    RunnablePassthrough.assign(context=lambda input: input["context"])
    | rag_prompt
    | model
    | StrOutputParser()
)

answer = chain.invoke({"context": '', "question": 'что такое нефть?'})
print(answer)