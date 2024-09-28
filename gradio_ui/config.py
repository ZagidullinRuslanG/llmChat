from llama_index.legacy import PromptTemplate
from llama_index.legacy.chat_engine.types import ChatMode
from llama_index.legacy.response_synthesizers.type import ResponseMode

query_wrapper_prompt = PromptTemplate(
    "GPT4 Correct User: {query_str}<|end_of_turn|>GPT4 Correct Assistant: "
) # openchat

class Config:

    # Chroma
    CHROMA_PATH = r"gradio_ui\chroma"


    # MODEL
    MODEL_PATH = r"C:\Work\Gazprom\LLM\llmChat\data\weights\openchat_3.5.Q5_K_M.gguf"
    SECOND_MODEL_PATH = r"C:\Work\Gazprom\LLM\llmChat\data\weights\starling-lm-7b-alpha.Q5_K_M.gguf"
    N_CTX = 16384 // 2
    N_BATCH = 9000 // 2
    # N_BATCH = 512
    N_CTX_PROMPT = 6000  // 2
    NODE_CTX = 256
    
    # RETRIEVER
    SIMILARITY_TOP_K = 10
    SIMILARITY_CUTOFF = 0.79
    EXCLUDE_METADATA_KEYS = ["file_path", "file_type", "file_size", "creation_date", "last_modified_date", "last_accessed_date"]
    USE_GLOSSARY = True
    CHUNK_OVERLAP = 20
    MAX_NEW_TOKENS = 2190
    TEMPERATURE = 0.4
    N_GPU_LAYERS = -1

    # EMDEDDINGS
    EMBEDDING_MODEL_PATH = r"C:\Work\Gazprom\LLM\llmChat\data\weights\intfloat_multilingual-e5-large"
    EMBEDDING_MODEL_KWARGS = {"device": "cuda:0"}
    SCORE_THRESHOLD_MAX = 1

    # QUERY ENGINE
    STREAMING = True
    MODE = ResponseMode.SIMPLE_SUMMARIZE
  
    # DEBUG
    DEBUG               = True
    DEBUG_QUERY         = True

    # SERVICE
    DATA_DIR = "./data"
    INDEX_PATH = "./storage"
    SERVER_NAME = 'localhost'
    SERVER_PORT = 7860
    USERS = [('user', 'user_pass'), ('dev', 'dev_pass')]
    AVAILABLE_CHAT_TYPES = [ChatMode.CONDENSE_PLUS_CONTEXT,
                            ResponseMode.SIMPLE_SUMMARIZE,
                            ]