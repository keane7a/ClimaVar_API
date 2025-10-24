from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def get_embeddings(texts, model, batch_size=256, device=DEVICE):
    """
    Encode texts with Sentence Transformers on GPU (if available).
    Returns L2-normalized float32 numpy array (N, D).
    """
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (ST)"):
        batch = texts[i : i + batch_size]
        # convert_to_tensor=True gives us a torch tensor directly on device
        embs = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,  # cosine-ready
        )
        vecs.append(embs.detach().cpu().to(torch.float32).numpy())
    return np.vstack(vecs)


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
        st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
        )
        # return self.openai_client.embedding.create(input=[text], model=self.embedding_model).data[0].embedding
        return get_embeddings(text, st_model)

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
