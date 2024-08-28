import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
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

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings",
)

client.reset()

# client.delete_collection('docs')

client.get_or_create_collection('docs', embedding_function=ef)
print(client.get_collection('docs').get(where={'language': 'en'}))

print('Chroma client created')

