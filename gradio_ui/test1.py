
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LTImage
from pdfminer.image import ImageWriter
from PIL import Image
import os

from pdfminer.pdfcolor import LITERAL_DEVICE_CMYK
from pdfminer.pdfcolor import LITERAL_DEVICE_GRAY
from pdfminer.pdfcolor import LITERAL_DEVICE_RGB
from pdfminer.pdftypes import (
    LITERALS_DCT_DECODE,
    LITERALS_JBIG2_DECODE,
    LITERALS_JPX_DECODE,
    LITERALS_FLATE_DECODE,
)


url = r'C:\Work\Gazprom\LLM\llmChat\data\original_docx\bot\Кандидаты.pdf'

elements = []

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
        # img_index = 0
        # while os.path.exists(path):
        #     name = "%s.%d%s" % (image.name, img_index, ext)
        #     path = os.path.join(self.outdir, name)
        #     img_index += 1
        return name, path

doc_name = 'Кандидаты'
page_number = 7
PICTURE_FOLDER = r'C:/Work/Gazprom/LLM/llmChat/gradio_ui/pictures/'

iw = ImageWriter_named(PICTURE_FOLDER)


for page_layout in extract_pages(url, page_numbers=[page_number]):
    for element_ind, element in enumerate(page_layout):
        # print(element)
        if isinstance(element, LTTextContainer):
            # print(element.get_text())
            # print(element.bbox)
            elements.append(element)

        if isinstance(element, LTFigure):
            # print(element.bbox)
            elements.append(element)

s_elements = sorted(elements, key = lambda x: x.bbox[3], reverse=True)

for element in s_elements:
    
    if isinstance(element, LTTextContainer):
        # print(element)
        current_text = element.get_text().strip()

        if (len(current_text) <= 0):
            continue

        print(element.get_text(), end = '')
    
    elif isinstance(element, LTFigure):
        # print('PICTURE', element)
        for sub_element_ind, sub_element in enumerate(element):
            if not isinstance(sub_element, LTImage):
                continue

            iw.set_prefix(doc_name, page_number, element_ind, sub_element_ind)
            
            iw.export_image(sub_element)
            image_full_name = iw.get_current_full_name()

            print('Picture:', image_full_name)
    
    else:
        print(element)


