import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.src.losses import cosine_similarity
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import h5py


class ImageClassifiers(MRJob):
    def __init__(self, input_emb, args=None):
        super().__init__(args)
        self.input_emb = input_emb

    def steps(self):
        return [
            MRStep(mapper=self.mapper_csv_to_embeddings),
            MRStep(mapper=self.mapper_embeddings_to_scores),
            MRStep(reducer=self.reduce_max_score)
        ]

    def mapper_csv_to_embeddings(self, file_name):
        with h5py.File(f"images/{file_name}.hdf5", 'r') as f:
            csv_embeddings = f["images"][:]
            print(file_name)
            yield file_name, csv_embeddings

    def mapper_embeddings_to_scores(self, file_name, embeddings):
        scores = []
        for emb in embeddings:
            score = cosine_similarity(self.input_emb, emb)
            scores.append(score)
        yield file_name, scores


    def reduce_max_score(self, file_name, scores):
        max_score = np.max(scores)
        yield file_name, max_score


if __name__ == '__main__':
    ImageClassifiers.run()
