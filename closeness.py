from sentence_transformers import SentenceTransformer
import numpy as np
import math

def closeness(previous_annotations, new_annotation):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings = model.encode(previous_annotations)
    center = np.mean(embeddings, axis=0)

    new_embedding = model.encode(new_annotation)
    distance = math.dist(new_embedding, center)
    closeness_score = distance

    return closeness_score
    
previous_annotations = [
    "a dog",
    "a big brown dog",
    "a canine chasing a ball",
    "a big red ball",
    "a bouncing ball",
    "dog chasing",
    "dog walking",
    "a big dog running"
]

new_annotation = "something completely ridiculous"
print(new_annotation)
print("Closeness score: ", closeness(previous_annotations, new_annotation))
