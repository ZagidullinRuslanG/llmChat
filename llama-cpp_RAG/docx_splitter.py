from docx import Document as DocX
from langchain_core.documents import Document as ChromaDocument

def reset_paragraph_dict():
    return {'header': 'None', 'body': []}

def name_to_id(doc_name, ind):
    return f'{doc_name}_{ind}'


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

        current_doc = ChromaDocument(
            page_content=full_text,
            metadata = {'source': 'document'},
            id = current_id
        )

        docs_list.append(current_doc)
        ids_list.append(current_id)

    return docs_list, ids_list


    # for pl in paragraph_list:
    #     print(pl['header'])

    #     for body in pl['body']:
    #         print(f'\t{body[:32]}')

if __name__ == '__main__':
    # doc_path = r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\cand.docx'

    doc_path = r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\event.docx'

    docs, ind = split_doc_from_headers(doc_path)

    for doc in docs:
        print(doc)


