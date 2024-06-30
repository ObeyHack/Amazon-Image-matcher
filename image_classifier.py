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

    def steps(self):
        return [
            MRStep(mapper=self.mapper_raw),
            MRStep(mapper=self.mapper_embeddings_to_scores),
            MRStep(reducer=self.reduce_max_score)
        ]

    def mapper_raw(self, wet_path, wet_uri):
        with h5py.File(f"{wet_path}.hdf5", 'r') as f:
            csv_embeddings = f["images"][:]
            print(wet_path)
            yield wet_path, csv_embeddings

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
