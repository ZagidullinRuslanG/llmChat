from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from get_context import get_context_text


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

        model = ChatOllama(
            model="rusmodel",
            callbacks = callbacks, 
            temperature=0.2, 
            # num_predict = 128
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
        results = get_context_text(question, k = 5)

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
