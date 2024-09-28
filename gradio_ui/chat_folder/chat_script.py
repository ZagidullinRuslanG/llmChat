from operator import itemgetter
import os
from config import Config as cfg
from prompts.default_prompts import DEFAULT_SYSTEM_PROMPT, GLOSSARY
import torch

from langchain_community.llms import LlamaCpp

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOllama

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import gradio as gr

from embed_folder.embed_script import *


SCORE_THRESHOLD_MAX = cfg.SCORE_THRESHOLD_MAX

model_path = r'C:\Work\Gazprom\LLM\llmChat\data\weights\starling-lm-7b-alpha.Q5_K_M.gguf'

# Initialize chat model
# llm = LlamaCpp(
#     model_path=model_path, 
#     temperature=0.5,
#     max_new_tokens=1000,
#     context_window=16379-1000,
#     generate_kwargs={},
#     n_ctx=8192,
#     n_gpu_layers=-1, 
#     n_threads=6, 
#     n_batch=512, 
#     verbose=True
# )

llm = ChatOllama(
    model="llama3.1",
    temperature=0.3, 
    # num_predict = 128,
)


# template = """You are senior python developer with great expirience in gradio and LLM."""

template = """You are helpfull assistant."""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = (chat_prompt | llm)

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)


embed_context_log = "None"
prompt_log = "None"


def stream_response(*args):

    input, history, temperature, sys_prompt, glossary, n_nodes_ctx = args

    global embed_context_log

    full_context = get_context_text(input, n_nodes_ctx, SCORE_THRESHOLD_MAX)

    embed_context_log = format_context_to_log(full_context)

    if input is not None:
        partial_message = '<img src="https://images.unsplash.com/photo-1727199204795-9607950eff99?q=80&w=2563&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D">\n'
        # ChatInterface struggles with rendering stream
        for response in chain_with_message_history.stream({"input": input}, {"configurable": {"session_id": "unused"}}):

            response_chunk = ""

            if type(response) is str:
                response_chunk = response
            else:
                response_chunk = response.content

            partial_message += response_chunk

            # print(response_chunk, end = '')

            yield partial_message


def is_user_dev(user):
    return True


def update_embed_log(timer):
    
    return embed_context_log, embed_context_log