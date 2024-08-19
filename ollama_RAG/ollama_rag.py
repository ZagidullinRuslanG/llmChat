from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Класс взаимодействия с локальной LLM
class local_llm:
    
    # Инициализация начальных параметров
    def __init__(self) -> None:
        rag_template: str = (
            'Давай краткие и точные ответы в соответствии с заданным контекстом.\n'
            'Если в представленной информации нет ответа на вопрос, просто ответь, что не знаешь.\n'
            'Используй максимум 3 предложения и соблюдай краткость.\n'
            '<context>\n'
            '{context}\n'
            '</context>\n'
            '\n'
            '{question}'
        )

        rag_prompt = ChatPromptTemplate.from_template(rag_template)

        callbacks = [StreamingStdOutCallbackHandler()]

        model = ChatOllama(
            model="rusmodel",
            callbacks = callbacks, 
            temperature=0.25, 
            # num_predict = 128
        )

        self.chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | model
            | StrOutputParser()
        )
    
    # Создание эмбеддингов из URL адреса
    def embed_from_url(self, url: str) -> None:

        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(data)
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    # Запрос к модели
    def ask_question(self, question: str) -> None:

        docs = self.vectorstore.similarity_search(question)
        
        print(f'Вопрос:\n{question}\nОтвет:')
        answer = self.chain.invoke({"context": docs, "question": question})
        print('\n')

        return answer


if __name__ == '__main__':


    llm = local_llm()

    # Ссылка на вики по Газпром нефть
    llm.embed_from_url(r"https://ru.wikipedia.org/wiki/%D0%93%D0%B0%D0%B7%D0%BF%D1%80%D0%BE%D0%BC_%D0%BD%D0%B5%D1%84%D1%82%D1%8C")

    # Ссылка на вики по Москве  
    llm.embed_from_url(r'https://ru.wikipedia.org/wiki/%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B0')


    llm.ask_question("Когда была основана компания Газпромнефть?")
    llm.ask_question("Чем занимается многопрофильная компания NIS?")

    llm.ask_question("Какая протяженность метро в Москве?")
    llm.ask_question("Как обстоят дела с наукой в Москве?")

