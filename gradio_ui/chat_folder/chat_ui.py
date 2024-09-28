from operator import itemgetter
import os
from config import Config as cfg
from prompts.default_prompts import DEFAULT_SYSTEM_PROMPT, GLOSSARY
import torch
from chat_folder.chat_script import *

import gradio as gr


def get_chat_ui():

    msg_box = gr.Textbox(container=False, show_label=False,
                            label="Message",
                            placeholder="Введите сообщение...",
                            scale=7,
                            autofocus=True,
                            )

    with gr.Blocks(fill_height=True) as block:
        with gr.Tabs() as tabs:
            with gr.Row():
                with gr.Column(scale=1):
                    header = f"""
                                **LLM**: {cfg.MODEL_PATH.split("/")[-1]}
                                **cuda**: {torch.cuda.is_available()}
                                **gpu**: {torch.cuda.get_device_name(0)}
                                """
                    gr.Markdown(header)
                #     allocMem = gr.Markdown()
                # with gr.Column(scale=1):
                #     button = gr.Button(value="Проверить видеопамять" , size="sm" , scale=1)
                #     button.click(getAllocMem, outputs=[allocMem])
            with gr.Row():
                with gr.TabItem("Настройки"):
                    update_config_btn = gr.Button(value='Обновить конфиг')
                    update_config_btn.click()

                    is_dev_md = gr.Markdown(visible=False)
                    sys_prompt = gr.Textbox(label='Системный промпт', value=DEFAULT_SYSTEM_PROMPT)
                    glossary = gr.Textbox(label='Словарь', value=GLOSSARY)
                    temp = gr.Slider(minimum=0, maximum=1, step=0.05, value=cfg.TEMPERATURE, label='Температура')
                    with gr.Blocks(visible=False) as optional:
                        n_nodes_ctx = gr.Slider(minimum=1, maximum=cfg.SIMILARITY_TOP_K+20, step=1, value=cfg.SIMILARITY_TOP_K, label='Количество нод контекста')
                        nodes_help_txt = f"""
                        При использовании режима **Simple Summarize** количество нод контекста можно увеличить до 23. \n
                        При слишком большом количестве нод контекста, возможно возникновение ошибки (контекст модели ограничен).
        """
                        nodes_help_md = gr.Markdown(nodes_help_txt)
                        def change_visible(request: gr.Request):
                            user = request.username
                            return (gr.Slider(minimum=0, maximum=cfg.SIMILARITY_TOP_K+20, step=1, value=cfg.SIMILARITY_TOP_K, label='Количество нод контекста', visible=is_user_dev(user)),
                                    gr.Markdown(nodes_help_txt, visible=is_user_dev(user)))
                        block.load(change_visible, inputs=None, outputs=[n_nodes_ctx, nodes_help_md])

                with gr.TabItem("Чат"):

                    with gr.Blocks(fill_width=True, css=".block  .svelte-12cmxck { height:calc(100vh - 380px)!important; }"):
                    
                        chat = gr.ChatInterface(fn = stream_response, #additional_outputs=[nodes, res_prompt], # only with custom gradio
                            additional_inputs=[temp, sys_prompt, glossary, n_nodes_ctx],
                            title='ERACHAT',
                            submit_btn='Отправить',
                            retry_btn='🔄  Перегенерировать',
                            undo_btn= '🔙  Отменить',
                            clear_btn='🗑️  Очистить',
                            textbox=msg_box,
                            fill_height= True,
                            examples=[['Что такое бригада?'], ['Что такое скважина?']])
                    
                    
                with gr.TabItem("Context"):
                    embed_timer = gr.Timer(1)

                    embed_log = gr.TextArea(value="Embedding text will be here", label='Context')
                    prompt_log = gr.TextArea(value="Your prompt text will be here", label='Prompt')
                    embed_timer.tick(fn = update_embed_log, inputs=[embed_timer], outputs=[embed_log, prompt_log])


            
        return block
