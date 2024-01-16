import time

from sentence_transformers import SentenceTransformer
sentences = ["open mailbox", "unlock case", "go east", "You see a locked case", "You are south of a house"]

models = [
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L6-v2',
    'BAAI/bge-small-en-v1.5',
    'TaylorAI/gte-tiny',
]

for model_name in models:
    model = SentenceTransformer(model_name)

    t0 = time.time()
    embeddings = model.encode(sentences)
    t1 = time.time()

    print(f"Model: {model_name} in {(t1 - t0) * 1000.0} msec")
