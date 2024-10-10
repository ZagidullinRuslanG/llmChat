from operator import itemgetter
import os
from config import Config as cfg
from prompts.default_prompts import DEFAULT_SYSTEM_PROMPT, GLOSSARY
import torch
from chat_folder.chat_script import *

import gradio as gr
from chat_folder.terminal_parse import get_ollama_model_list


def get_chat_ui():


    msg_box = gr.Textbox(container=False, show_label=False,
        label="Message",
        placeholder="Введите сообщение...",
        scale=7,
        autofocus=True,
        )

    with gr.Blocks(fill_height=True, css=".block  .svelte-12cmxck { height:calc(100vh - 380px)!important; }") as block:

        #     allocMem = gr.Markdown()
        # with gr.Column(scale=1):
        #     button = gr.Button(value="Проверить видеопамять" , size="sm" , scale=1)
        #     button.click(getAllocMem, outputs=[allocMem])

        with gr.Accordion("Настройки", open=False):
            

            sys_prompt = gr.Textbox(label='Системный промпт', value=DEFAULT_SYSTEM_PROMPT)
            glossary = gr.Textbox(label='Словарь', value=GLOSSARY)
            temp = gr.Slider(minimum=0, maximum=1, step=0.05, value=cfg.TEMPERATURE, label='Температура')

            n_ctx = gr.Slider(minimum=1024, maximum=1024*126, step=1024, value=cfg.N_CTX, label='Размер окна контекста')

            n_nodes_ctx = gr.Slider(minimum=0, maximum=cfg.SIMILARITY_TOP_K+20, step=1, value=cfg.SIMILARITY_TOP_K, label='Количество нод контекста')
            max_threshold_ctx = gr.Slider(minimum=0, maximum=1, step=0.01, value=cfg.SIMILARITY_CUTOFF, label='Максимальное несоответствие контекста')

            model_list, starting_model = get_ollama_model_list(in_one_line=True, find_starting=cfg.START_MODEL_NAME)
            print(model_list, starting_model)
            model_dropdown = gr.Dropdown(model_list, label='Выбор модели', value = starting_model, interactive=True)

            update_config_btn = gr.Button(value='Обновить конфиг')
            update_config_btn.click(fn = update_model_from_config, inputs=[
                model_dropdown,
                sys_prompt, glossary, temp, n_nodes_ctx, max_threshold_ctx, n_ctx
                ])
            

            flush_cache_btn = gr.Button(value = 'Очистить текущий VRAM')
            flush_cache_btn.click(fn = flush_VRAM)

            stop_ollama_model_btn = gr.Button(value = 'Остановить модель Ollama')
            stop_ollama_model_btn.click(stop_ollama_model)

            flush_VRAM_after_use = gr.Checkbox(value=True, label='Очищать VRAM после запроса за эмбеддингами')
            flush_VRAM_after_use.change(fn = change_flush_after_use, inputs=[flush_VRAM_after_use])

            preload_ollama_model_btn = gr.Button(value='Предзагрузка модели')
            preload_ollama_model_btn.click(fn = preload_ollama_model)

            # slider
            

        with gr.Accordion("Контекст", open=False):
            embed_timer = gr.Timer(1)

            embed_log = gr.TextArea(value="Embedding text will be here", label='Context')
            prompt_log = gr.TextArea(value="Your prompt text will be here", label='Prompt')
            embed_timer.tick(fn = update_embed_log, inputs=[embed_timer], outputs=[embed_log, prompt_log])
            
        

        # TODO: str кнопки не очищают историю бота, сделать кастомные кнопки

        chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=800,
                )

        chat = gr.ChatInterface(fn = stream_response, #additional_outputs=[nodes, res_prompt], # only with custom gradio
            additional_inputs=[temp, sys_prompt, glossary, n_nodes_ctx, max_threshold_ctx],
            title='ERACHAT',
            chatbot=chatbot,
            submit_btn='Отправить',
            retry_btn='🔄  Перегенерировать',
            undo_btn= '🔙  Отменить',
            clear_btn='🗑️  Очистить',
            textbox = msg_box,
            examples=[['Что такое бригада?'], ['Что такое скважина?'], ['Сократи ответ'], ['Что такое скважина кандидат? Приведи инструкцию с изображениями, как ее добавить.'], ['Приведи инструкцию с изображениями, как редактировать скважину кандидата.']])

        chat.load(fn = update_model_from_config, inputs=[
            model_dropdown,
            sys_prompt, glossary, temp, n_nodes_ctx, max_threshold_ctx, n_ctx
            ])
        

        header_log = gr.Markdown(chat_update())

        header_log_timer = gr.Timer(1)
        header_log_timer.tick(fn = chat_update, outputs=[header_log])
            
        # chat.chatbot.change(fn = chat_update, outputs=[header_log])
            
        return block
