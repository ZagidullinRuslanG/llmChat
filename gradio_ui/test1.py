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


store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]


history = get_session_history("1", "1")
history.add_message(AIMessage(content="hello"))
print(store)


def rag_retriever(query):
    assert isinstance(query, str)
    full_context = get_context_text(query, 3, 1)
    print(full_context)
    return [doc for doc, _ in full_context]



rag_retriever = RunnableLambda(rag_retriever)

print(rag_retriever.invoke("hello"))

model = ChatOllama(
    model="llama3.1",
    temperature=0.3, 
    # num_predict = 128,
    verbose=True
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""Ты — чатбот техподдержка, специализирующийся на {ability}.
Детально изучи контекст, чтобы понять, какие данные нужно использовать.
Если запрос нечеткий или требует уточнения, попроси пользователя уточнить вопрос или предложи возможные направления ответа.
Всегда старайся предоставлять наиболее полезную и точную информацию, основываясь на заданном контексте.
В самом начале контекста находится максимально полезная информация, чем дальше, тем менее полезная.
'<context>'
'{context}'
'</context>
'""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


context = itemgetter("question") | rag_retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
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

# print(context.invoke({"question": "hello", "other_stuff": "ignored"}))

# print(first_step.invoke({"question": "hello", "history": "boom"}))

# print(chain.invoke({"question": "hello", "history": [], "ability": "math"}))

# print(
#     with_message_history.invoke(
#         {"ability": "math", "question": "What does cosine mean?"},
#         config={
#             "configurable": {"user_id": "user_id", "conversation_id": "conversation_id"}
#         },
        
#     )
# )

while (True):
    inp = input()

    for response in with_message_history.stream({"ability": "ремонте нефтяных и газовых скважин", "question": inp},
            config={
                "configurable": {"user_id": "user_id", "conversation_id": "conversation_id"}
            }):

                response_chunk = ""

                if type(response) is str:
                    response_chunk = response
                else:
                    response_chunk = response.content

                print(response_chunk, end = '')
    
    print()

# print(store)