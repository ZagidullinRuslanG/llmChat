import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')

if os.path.exists(CHROMA_PATH):
    print('Already exists!')

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
    
)

# print(client.get_collection('docs').get())

client.reset()

# client.delete_collection('docs')

client.get_or_create_collection('docs')
print(client.get_collection('docs').get(where={'language': 'en'}))

print('Chroma client created')

