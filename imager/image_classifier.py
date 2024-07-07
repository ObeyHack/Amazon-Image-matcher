import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from mrjob.protocol import RawValueProtocol
from keras.src.losses import cosine_similarity
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import h5py


class ImageClassifiers(MRJob):
    INPUT_PROTOCOL = RawValueProtocol

    def __init__(self, input_emb, args=None):
        super().__init__(args)
        self.input_emb = input_emb

    def mapper(self, _, input_path):
        print(input_path)
        input_path, csv_embeddings = self.mapper_file_to_embeddings(input_path)
        file_name, scores = self.mapper_embeddings_to_scores(input_path, csv_embeddings)
        yield file_name, scores

    def mapper_file_to_embeddings(self, file_name):
        with h5py.File(f"images/{file_name}.hdf5", 'r') as f:
            csv_embeddings = f["images"][:]
            print(file_name)
            return file_name, csv_embeddings

    def mapper_embeddings_to_scores(self, file_name, embeddings):
        scores = []
        for emb in embeddings:
            score = cosine_similarity(self.input_emb, emb)
            scores.append(score)
        return file_name, scores

    def reducer(self, file_name, scores):
        max_score = np.max(scores)
        argmax_score = np.argmax(scores)
        yield file_name, max_score, argmax_score


if __name__ == '__main__':
    ImageClassifiers.run()
