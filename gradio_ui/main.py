import gradio as gr
from config import Config as cfg

from chat_folder.chat_ui import get_chat_ui
from embed_folder.embed_ui import get_embed_ui

ui_pages, ui_pages_titles = [], []

ui_pages.append(get_chat_ui())
ui_pages_titles.append('Чат')

ui_pages.append(get_embed_ui())
ui_pages_titles.append('Эмбеддинги')


demo = gr.TabbedInterface(ui_pages, ui_pages_titles, css=".block  .svelte-12cmxck { height:calc(100vh - 380px)!important; }")
demo.launch(server_name=cfg.SERVER_NAME, server_port=cfg.SERVER_PORT, auth=cfg.USERS)