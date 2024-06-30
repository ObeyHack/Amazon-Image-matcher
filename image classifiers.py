import h5py
import numpy as np
from progress_executor.main import executor
from tqdm.auto import tqdm
import glob
from VGG import get_image_embeddings, load_image_bytes, load_image, get_score, label
import os
from mrjob.job import MRJob
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


def argmax_score(input_embedding, embeddings):
    return np.argmax([get_score(input_embedding, emb) for emb in embeddings])


def max_csv(input):
    path = os.getcwd() + "/Image"
    csv_files = glob.glob(os.path.join(path, "*.hdf5"))

    intput_embedding = get_image_embeddings(load_image(input))
    csv_scores = {}
    for file in tqdm(csv_files):
        with h5py.File(f"images/{file}.hdf5", "r") as f:
            if "images" in f:
                csv_embeddings = f["images"]
                csv_scores[file] = executor.submit(argmax_score, intput_embedding, csv_embeddings)

    return max(csv_scores, key=csv_scores.get)



if __name__ == '__main__':
    img = load_image("sunflower/sunflower1.jpg")
    print(max_csv(img))
