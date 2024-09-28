from embed_folder.embed_script import *

import gradio as gr
import numpy as np
import pandas as pd


def get_selected_from_df(selected_index: gr.SelectData, dataframe_origin):
    val = dataframe_origin.iloc[selected_index.index[0]]

    print(selected_index.index[0], val[0])

    # return dataframe_origin, dataframe_target       


def get_embed_ui():

    with gr.Blocks(fill_height=True) as ui:
        file_input = gr.File(label="Загрузить документы", file_count = "multiple", file_types=['docx'])
        print_button = gr.Button("Показать разделение")
        uploaded_files = gr.Textbox(value='Разделение документов')
        print_button.click(update_output, inputs=[file_input], outputs=[uploaded_files])


        embed_button = gr.Button('Вычислить эмбеддинги')
        clear_vectorstore_button = gr.Button('Очистить БД')


        embed_button.click(add_documents, inputs=[file_input])
        clear_vectorstore_button.click(reset_and_create_chroma_client)


        gr_df = gr.DataFrame(
            # value=pd.DataFrame([[np.random.randint(2, 1024) * 'A'] * 4] * 10, columns=list('ABCD')),
            value = get_df(),
            # wrap=True,
            height = 500,
            interactive= False
        )

        gr_df.select(get_selected_from_df, inputs = [gr_df], outputs = None)


        gr.Button("Test")


        return ui