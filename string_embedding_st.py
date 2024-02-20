import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Embedding leaderboard: https://huggingface.co/spaces/mteb/leaderboard

# script to use sentence transformers to embed two company names and see if they are similar

revision = None  # Replace with the specific revision to ensure reproducibility in  case the model is updated.

model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)

def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def compute_similarity(embeddings):
    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
    print(similarity.cpu().numpy())
    return similarity