from embed_folder.embed_script import *
from time import sleep
import torch

import gc


def flush_memory():

    gc.collect()
    torch.cuda.empty_cache()

class Embedder(object):
    def __init__(self, file_name):
        self.file_name = file_name
    
    def __enter__(self):
        # self.file = open(self.file_name, 'w')

        add_document(self.file_name)

        

    def __exit__(self, *args):
        # self.file.close()
        pass

print('Adding')
# with Embedder(r"C:\Work\Gazprom\LLM\llmChat\data\original_docx\avr.docx"):
#     pass

add_document(r"C:\Work\Gazprom\LLM\llmChat\data\original_docx\avr.docx")
flush_memory()

print('Added')

sleep(10)