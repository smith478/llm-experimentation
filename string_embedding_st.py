import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Embedding leaderboard: https://huggingface.co/spaces/mteb/leaderboard

revision = None  # Replace with the specific revision to ensure reproducibility in  case the model is updated.

model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)

def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def compute_similarity(embeddings):
    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
    print(similarity.cpu().numpy())
    return similarity

if __name__ == '__main__':
    texts = ["Apple Inc.", "Microsoft Corporation"]
    embeddings = compute_embeddings(texts)
    similarity = compute_similarity(embeddings)
    print(similarity)