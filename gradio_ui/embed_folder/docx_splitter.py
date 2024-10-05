from docx import Document as DocX
from langchain_core.documents import Document as ChromaDocument

import cv2
import numpy as np
import os
from config import Config as cfg
from uuid import uuid4

def reset_paragraph_dict():
    return {'header': 'None', 'body': []}

def name_to_id(doc_name, ind):
    # return f'{doc_name}_{ind}'
    return str(uuid4())


def split_doc_from_headers(doc_path):

    document = DocX(doc_path)

    paragraph_list = []

    current_paragraph = reset_paragraph_dict()

    for par in document.paragraphs:
        # print(par.style, par.text)

        if(par.style.name.startswith('Heading')):

            paragraph_list.append(current_paragraph)
            current_paragraph = reset_paragraph_dict()
            
            current_paragraph['header'] = par.text
            continue

        current_paragraph['body'].append(par.text)


    paragraph_list.append(current_paragraph)


    paragraph_list = [pl for pl in paragraph_list if len(pl['body']) > 0]

    docs_list = []
    ids_list = []

    for ind, paragraph in enumerate(paragraph_list):
        full_text = paragraph['header'] + '\n'
        full_text += "\n".join(paragraph['body'])

        current_id = name_to_id(doc_path, ind)

        print(paragraph['header'])

        header_img = np.zeros(shape=(720,1280,3),dtype=np.uint8)
        header_img.fill(255)
        header_img = cv2.putText(header_img, paragraph['header'], (100, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)



        image_name = f'{current_id}.jpg'
        image_path = os.path.join(cfg.IMAGE_FOLDER, image_name)
        # print(image_path)

        cv2.imwrite(image_path, header_img)


        current_doc = ChromaDocument(
            page_content=full_text,
            metadata = {'source': 'document', 'image': image_name},
            id = current_id
        )

        docs_list.append(current_doc)
        ids_list.append(current_id)

    return docs_list, ids_list


if __name__ == '__main__':

    doc_path = r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\avr.docx'

    docs, ind = split_doc_from_headers(doc_path)

    for doc in docs:
        print(doc.metadata)


