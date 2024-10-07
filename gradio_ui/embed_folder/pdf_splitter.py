
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LTImage, LTChar, LTTextBox, LTTextLine
from pdfminer.image import ImageWriter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1
import os
import re
import tqdm
import json


url = r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\bot\Кандидаты.pdf'
PICTURE_FOLDER = r'C:/Work/Gazprom/LLM/llmChat/gradio_ui/pictures/'
doc_name = 'Кандидаты'


UPPER_BOUNDARY = 810
LOWER_BOUNDARY = 53

def get_pdf_number_of_pages(url) -> int:
    file = open(url, 'rb')
    parser = PDFParser(file)
    document = PDFDocument(parser)
    return resolve1(document.catalog['Pages'])['Count']

def out_of_page_boundary(element):
    """
    Выходит ли элемент за рассматриваемые рамки страницы
    """
    
    if not hasattr(element, 'bbox'):
        print('No bbox found')
        return True
    
    y = element.bbox[3]

    return y <= LOWER_BOUNDARY or y >= UPPER_BOUNDARY


def is_4th_list_indent(text_box: LTTextBox):
    """
    Найти строки, которые начинаются с "1.2.3.4", которые могут быть заголовками
    """

    text = text_box.get_text().strip()

    pattern = r'^\d\.\d\.\d\.\d\b'

    return True if re.match(pattern, text) else False

    # return True

def text_is_bold(text_box: LTTextBox, bold_ratio = 0.8):
    """
    Является ли шрифт данного текста жирным
    """

    charachters_n = 0
    bold_charachters = 0

    for text_line in text_box:

        if not isinstance(text_line, LTTextLine):
            continue

        for char_elem in text_line:

            if not isinstance(char_elem, LTChar):
                continue

            charachters_n += 1
            
            if not ('bold' in str(char_elem.fontname).lower()):
                continue

            bold_charachters += 1

    return bold_charachters / charachters_n >= bold_ratio



class ImageWriter_named(ImageWriter):

    name_prefix = 'test_prefix_'

    current_full_name = ''

    def set_prefix(self, doc_name, page_number, ind, sind):
        self.set_next_image_prefix(f'{doc_name}_P{page_number:03}_E{ind:03}_SE{sind:02}_')

    def set_next_image_prefix(self, prefix):
        self.name_prefix = prefix
    
    def get_image_full_name(self, image: LTImage, ext: str) -> str:
        name = self.name_prefix + image.name + ext

        self.current_full_name = name

        return name
    
    def get_current_full_name(self):
        return self.current_full_name


    def _create_unique_image_name(self, image: LTImage, ext: str):
        name = self.get_image_full_name(image, ext)

        path = os.path.join(self.outdir, name)
        return name, path

def parse_page_layout(page_layout, page_number: int, iw: ImageWriter_named, verbose = False):

    elements = []

    # iw = ImageWriter_named(PICTURE_FOLDER)

    # for page_layout in extract_pages(url, page_numbers=[page_number]):

    for element_ind, element in enumerate(page_layout):
        if isinstance(element, LTTextContainer):
            elements.append(element)

        if isinstance(element, LTFigure):
            elements.append(element)

    s_elements = sorted(elements, key = lambda x: x.bbox[3], reverse=True)

    json_elements = []

    for element in s_elements:

        if out_of_page_boundary(element):
            continue
        
        if isinstance(element, LTTextContainer):
            current_text = element.get_text().strip()

            if (len(current_text) <= 3):
                continue

            

            element_is_bold = text_is_bold(element)
            element_is_4th_list_indent = is_4th_list_indent(element)
            element_is_header = element_is_bold or element_is_4th_list_indent

            if verbose:
                print(current_text)
                if element_is_header:
                    print('*'*50)

            json_elements.append({
                'type': 'text', 
                'content': current_text,
                'is_header': element_is_header,
                'page_number': page_number,
            })

        
        elif isinstance(element, LTFigure):
            for sub_element_ind, sub_element in enumerate(element):
                if not isinstance(sub_element, LTImage):
                    continue

                iw.set_prefix(doc_name, page_number, element_ind, sub_element_ind)
                
                iw.export_image(sub_element)
                image_full_name = iw.get_current_full_name()

                if verbose:
                    print('Picture:', image_full_name)

                json_elements.append({
                    'type': 'image',
                    'path': image_full_name,
                    'page_number': page_number,
                })


    return json_elements

# def join_pdf_data(data):

#     all_elements = []
#     current_element = []


#     current_content_block = []
#     last_page_number_start = None
#     last_page_number_end = None

#     current_text_block = []

#     for element in data:

#         if element['type'] == 'image':

#             # current_content_block.append({'type': 'text', 'content': current_text_block})
#             current_content_block.append(current_text_block)
#             current_content_block.append(element)

#             current_text_block = []
#             continue

#         if (element['is_header']):

#             current_content_block.append(current_text_block)
#             current_element.append(current_content_block)
#             all_elements.append(current_element)
    
#             current_element = []
#             current_content_block = []
#             current_text_block = []

#             current_content_block.append({
#                 'type': 'header', 
#                 'content': element
#             })

#             continue

#         # current_text_block += '\n' + element['content']
#         current_text_block.append(element)
    


#     all_elements.append(current_element)

    
#     for elem in all_elements:
#         print(elem)


#     return


# def join_pdf_data(data):

#     all_elements = []

#     current_text_block = []

#     current_content_block = []

#     for element in data:

#         if element['type'] == 'image':

#             # current_content_block.append({'type': 'text', 'content': current_text_block})
#             current_content_block.append({'type': 'text_block', 'content': current_text_block})
#             current_content_block.append(element)

#             current_text_block = []
#             continue

#         if (element['is_header']):

#             current_content_block.append({'type': 'text_block', 'content': current_text_block})
            
#             contents = []

#             content_block = {
#                 'header': current_content_block[0]['content'],
#                 'content': [ccb for ccb in current_content_block[1:]]
#             }

#             all_elements.append(content_block)
    
#             current_content_block = []
#             current_text_block = []

#             current_content_block.append(element)

#             continue

#         # current_text_block += '\n' + element['content']
#         current_text_block.append(element)
    


#     all_elements.append(current_content_block)

    
#     for elem in all_elements[:5]:
#         print(elem)
#         print()


#     return

def join_content_from_block(block):
    output_str = '\n'.join([ctb['content'].replace('  ', ' ') for ctb in block])
    return output_str


def get_last_page_from_block(block, output_else = None):
    arr = [int(ctb['page_number']) for ctb in block]

    if len(arr) <= 0:
        return output_else
    
    return max(arr)


def join_pdf_data(data):

    all_elements = []

    current_text_block = []

    current_content_block = {'header': None, 'content': [], 'first_page_number': None, 'last_page_number': None}

    data_N = len(data)

    for element_ind, element in enumerate(data):

        if element['type'] == 'image':

            joined_text = join_content_from_block(current_text_block)
            if (len(joined_text) > 0):
                current_content_block['content'].append({
                    'type':'text',
                    'content': joined_text,
                    'page_number': get_last_page_from_block(current_text_block, None)
                })

            current_content_block['content'].append(element)

            current_text_block = []
            continue

        if (element['is_header'] or element_ind == data_N - 1):

            joined_text = join_content_from_block(current_text_block)
            if (len(joined_text) > 0):
                current_content_block['content'].append({
                    'type':'text',
                    'content': joined_text,
                    'page_number': get_last_page_from_block(current_text_block, None)
                })

            current_content_block['last_page_number'] = get_last_page_from_block(current_content_block['content'], current_content_block['first_page_number'])


            all_elements.append(current_content_block)
    
            current_content_block = {'header': None, 'content': [], 'first_page_number': None, 'last_page_number': None}
            current_text_block = []

            current_content_block['header'] = element['content']
            current_content_block['first_page_number'] = element['page_number']

            continue

        current_text_block.append(element)
    


    # all_elements.append(current_content_block)

    all_elements = all_elements[1:]

    
    # for elem in all_elements:
    #     print(elem)
    #     print()


    return all_elements

def parse_pdf_pages(pdf_path: str, skip_pages: list = []):

    iw = ImageWriter_named(PICTURE_FOLDER)

    total_pages = get_pdf_number_of_pages(pdf_path)

    pdf_data = []

    for page_ind, page_layout in enumerate(tqdm.tqdm(extract_pages(pdf_path), total=total_pages)):
        
        if (page_ind + 1) in skip_pages:
            continue

        page_data = parse_page_layout(page_layout, page_ind + 1, iw)

        pdf_data += page_data

    pdf_data = join_pdf_data(pdf_data)

    return pdf_data

if __name__ == '__main__':
    elems = parse_pdf_pages(url, [1, 2, 3, 4, 5])

    with open("gradio_ui/json_output/mydata.json", "w", encoding='utf-8') as output:
        json.dump(elems, output, indent=3, ensure_ascii=False)
