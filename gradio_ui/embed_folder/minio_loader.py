from minio import Minio
import os
from dotenv import load_dotenv
load_dotenv()
import tqdm

import gradio as gr

class MinioLoader:
    def __init__(self, verbose = False):
        self.client = Minio("localhost:9000",
            access_key=os.getenv('MINIO_ACCESS_KEY'),
            secret_key=os.getenv('MINIO_SECRET_KEY'),
            secure=False
        )

        self.bucket_name = 'bucket1'

        self.verbose = verbose

    
    def add_file(self, file_path, new_file_name):

        if self.verbose:
            print(f'Adding file {file_path} as {new_file_name}')

        self.client.fput_object(
            self.bucket_name,
            new_file_name,
            file_path)

        return
    
    def add_folder(self, folder_path):

        if not os.path.exists(folder_path):
            print(f'Folder {folder_path} not found')
            return
        

        ff_list = os.listdir(folder_path)

        
        for ff in tqdm.tqdm(ff_list, total = len(ff_list)):
            ff_path = os.path.join(folder_path, ff)

            if not os.path.isfile(ff_path):
                # Если это не файл а папка
                # self.add_folder(ff_path)
                continue
            
            self.add_file(ff_path, ff)

if __name__ == '__main__':
    mloader = MinioLoader()
    
    mloader.add_folder(r"C:\Work\Gazprom\LLM\llmChat\gradio_ui\pictures")

    

