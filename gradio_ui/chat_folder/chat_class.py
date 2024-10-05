from operator import itemgetter
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from embed_folder.embed_script import *

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


class LLMWithHistoryAndContext:
    def __init__(self, model_name, system_prompt, glossary, temperature, n_nodes_ctx, max_threshold_ctx):
        self.model = ChatOllama(
            model = model_name,
            temperature=temperature, 
            # num_predict = 128,
            verbose=True,
            keep_alive=-1,
            num_ctx = 10_000,
            
        )

        self.n_nodes_ctx = n_nodes_ctx
        self.max_threshold_ctx = max_threshold_ctx

        self.store = {}
        self.rag_retriever = RunnableLambda(self.rag_retriever)

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        # """Ты — чатбот техподдержка, специализирующийся на {ability}.
        # Детально изучи контекст, чтобы понять, какие данные нужно использовать.
        # Если запрос нечеткий или требует уточнения, попроси пользователя уточнить вопрос или предложи возможные направления ответа.
        # Всегда старайся предоставлять наиболее полезную и точную информацию, основываясь на заданном контексте.
        # В самом начале контекста находится максимально полезная информация, чем дальше, тем менее полезная.
        # '<context>'
        # '{context}'
        # '</context>
        # '""",
        #         ),
        #         MessagesPlaceholder(variable_name="history"),
        #         ("human", "{question}"),
        #     ]
        # )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
        system_prompt + """
        '<context>'
        '{context}'
        '</context>
        '""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        context = itemgetter("question") | self.rag_retriever | format_docs
        first_step = RunnablePassthrough.assign(context=context)
        chain = first_step | prompt | self.model

        self.invoker = RunnableWithMessageHistory(
            chain,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )



    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory()
        return self.store[(user_id, conversation_id)]


    def rag_retriever(self, query):
        assert isinstance(query, str)
        if not cfg.LOAD_EMBEDDINGS:
            return []
        
        if self.n_nodes_ctx <= 0:
            return []
        
        full_context = get_context_text(query, k = self.n_nodes_ctx, score_max_thresh = self.max_threshold_ctx)
        flush_VRAM()
        print(full_context)
        return [doc for doc, score in full_context]
    

    def stream(self, question: str, ability: str = "ремонте нефтяных и газовых скважин", user_id = "user_id", conversation_id = "conversation_id"):
        return self.invoker.stream(
            {"ability": ability, "question": question},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




