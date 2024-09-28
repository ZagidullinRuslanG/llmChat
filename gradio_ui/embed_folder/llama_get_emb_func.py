from typing import List

from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class LlamaCppEmbeddings_(LlamaCppEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text)[0] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)[0]
        return list(map(float, embedding))
    

def get_llama_cpp_embeddings():

    embed_model_path = r"C:\Work\Gazprom\LLM\llmChat\data\weights\intfloat_multilingual-e5-large"
    embed_model_path_kwargs = {"device": "cuda:0"}

    # embeddings = LlamaCppEmbeddings_(
    #     model_path=model_path,
    #     n_gpu_layers=-1
    # )

    embeddings = HuggingFaceEmbeddings(
        
        model_name = embed_model_path,
        model_kwargs=embed_model_path_kwargs
    )



    return embeddings
