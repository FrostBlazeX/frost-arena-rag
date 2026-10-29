from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love watching football at the pub.",
    "The best place to watch soccer is a sports bar.",
    "I enjoy baking bread on weekends.",
]
vectors = model.encode(sentences)
print("Vector shape:", vectors.shape)             # (3, 384)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("football vs soccer:", round(cosine(vectors[0], vectors[1]), 3))
print("football vs baking:", round(cosine(vectors[0], vectors[2]), 3))
