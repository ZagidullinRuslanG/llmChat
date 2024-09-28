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
from get_context import get_context_text
from llama_get_emb_func import get_llama_cpp_embeddings


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Класс взаимодействия с локальной LLM
class local_llm:
    
    # Инициализация начальных параметров
    def __init__(self) -> None:
        rag_template: str = (
            'Отвечай на поставленные вопросы только в рамках того контекста, что тебе предоставлен.\n'
            'Не оклоняйся от предоставленного контекста.\n'
            'Если в представленной информации нет ответа на вопрос, просто ответь, что не знаешь.\n'
            '<context>\n'
            '{context}\n'
            '</context>\n'
            '\n'
            '{question}'
        )

        rag_prompt = ChatPromptTemplate.from_template(rag_template)

        callbacks = [StreamingStdOutCallbackHandler()]

        model = LlamaCpp(
            model_path=r'C:\Work\Gazprom\LLM\llmChat\data\weights\starling-lm-7b-alpha.Q5_K_M.gguf', 
            temperature=0.2,
            n_gpu_layers=-1,
            n_ctx=2048,
            callbacks = callbacks
        )

        self.chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | model
            | StrOutputParser()
        )
    

    # Запрос к модели
    def ask_question(self, question: str) -> None:

        # docs = self.vectorstore.similarity_search(question)
        results = get_context_text(question, k = 2)

        print(f'{"-"*40}\nFound {len(results)} matching documents:\n')

        for res, score in results:
            print(f'Score: {score}\n{res}\n')

        docs = [res for res, score in results if score < 0.35]


        print(f'{"-"*40}\nUsing context from:\n')

        print(format_docs(docs))

        print(f'{"-"*40}\n')
        
        print(f'Вопрос:\n{question}\nОтвет:')
        answer = self.chain.invoke({"context": docs, "question": question})
        print('\n')

        return answer


if __name__ == '__main__':

    llm = local_llm()

    while(True):
        question = input("Ваш вопрос: ")
        llm.ask_question(question)

        # Как посмотреть график добычи?
        # Просмотр параметров по сважине в графике мероприятий
        # Как указать признаки для бригад ГРР?
