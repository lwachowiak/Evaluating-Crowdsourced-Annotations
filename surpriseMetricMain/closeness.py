import numpy as np
import math
from sentence_transformers import SentenceTransformer


class Closeness:

    def __init__(self, center) -> None:
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.center = center

    def initialise(self, previous_annotations):
        embeddings = self.model.encode(previous_annotations)
        self.center = np.mean(embeddings, axis=0)
        print(list(self.center))

    def score(self, new_annotation):
        new_embedding = self.model.encode(new_annotation)
        distance = math.dist(new_embedding, self.center)
        closeness_score = distance
        return closeness_score

    def getCenter(self):
        return self.center
