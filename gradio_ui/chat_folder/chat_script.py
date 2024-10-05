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
from chat_folder.chat_class import LLMWithHistoryAndContext
from chat_folder.terminal_parse import *

from time import perf_counter


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

# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0.3, 
#     # num_predict = 128,
# )


# template = """You are helpfull assistant."""

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", template),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ]
# )

# chain = (chat_prompt | llm)

# demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

# chain_with_message_history = RunnableWithMessageHistory(
#     chain,
#     lambda session_id: demo_ephemeral_chat_history_for_chain,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )

model = None

def update_model_from_config(
        model_dropdown: str, sys_prompt: str, glossary: str, 
        temp: float, n_nodes_ctx: int, max_threshold_ctx: int):


    model_name = model_dropdown.split(' / ')[0]
    
    global model
    model = LLMWithHistoryAndContext(
        model_name = model_name,
        system_prompt = sys_prompt,
        glossary = glossary,
        temperature = temp,
        n_nodes_ctx = n_nodes_ctx,
        max_threshold_ctx = max_threshold_ctx,
        )
    
    print(f"""Model updated with arguments:
        {model_dropdown=}
        {sys_prompt=}
        {glossary=}
        {temp=}
        {n_nodes_ctx=}
        {max_threshold_ctx=}""")


embed_context_log = "None"
prompt_log = "None"

token_generation_speed = 0.0
model_loading_time = 0.0


def stream_response(*args):

    start_model_loading = perf_counter()

    input_text, history, temperature, sys_prompt, glossary, n_nodes_ctx, max_threshold_ctx = args

    global embed_context_log, token_generation_speed, model_loading_time

    # full_context = get_context_text(input_text, n_nodes_ctx, max_threshold_ctx)

    # embed_context_log = format_context_to_log(full_context)

    # full_context = model.rag_retriever(input_text)
    # embed_context_log = format_context_to_log(full_context)

    # context_image = get_context_image(full_context)
    # print(context_image)

    if not (input_text is None):

        partial_message = ''

        # if not (context_image is None):
        #     partial_message += f'<img src="{context_image}">\n'

        time_start_generation = None

        response_stream = model.stream(input_text)

        n_tokens_generated = 0

        for response in response_stream:

            if time_start_generation is None:
                time_start_generation = perf_counter()
                model_loading_time = perf_counter() - start_model_loading

            response_chunk = ""

            if type(response) is str:
                response_chunk = response
            else:
                response_chunk = response.content

            partial_message += response_chunk

            # print(response_chunk, end = '')

            n_tokens_generated += 1

            token_generation_speed = n_tokens_generated / (perf_counter() - time_start_generation) 

            yield partial_message


def is_user_dev(user):
    return True


def update_embed_log(timer):
    
    return embed_context_log, embed_context_log


def chat_update():
    log_str = f'Model loading time: <b>{model_loading_time:.2f} second(s)</b><br>Generation speed: <b>{token_generation_speed:.3f} tokens / second</b><br>{get_ollama_loaded_status()}'

    return log_str