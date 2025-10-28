from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class EmbeddingModel:
    def __init__(
        self, embedding_model, openai_client, chromadb_client, collection_name, **kwargs
    ):
        self.embedding_model = embedding_model
        self.openai_client = openai_client
        self.chromadb_client = chromadb_client
        self.collection = self.chromadb_client.get_or_create_collection(
            name=collection_name
        )

    def _get_embeddings(self, text):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding

    def query_chromadb(self, query, top_n=3):
        query_embedding = self._get_embeddings(query)
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_n
        )
        res = []
        for i, metadata in enumerate(results["metadatas"][0]):
            metadata["text"] = results["documents"][0][i]
            res.append(metadata)
        return res
