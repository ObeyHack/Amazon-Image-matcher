import glob
import os
import numpy as np
import h5py
from keras.src.losses import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import pandas as pd

def mapper_file_to_embeddings(file_name):
    with h5py.File(f"images/{file_name}.hdf5", 'r') as f:
        if "images" not in f:
            return file_name, [0] * 512
        csv_embeddings = f["images"][:]
        return file_name, csv_embeddings

def mapper_embeddings_to_scores(file_name, embeddings, input_emb):
    scores = cosine_similarity(input_emb, embeddings)
    return file_name, scores

def reducer_scores_to_max(file_name, scores, top_k=5):
    max_scores = np.sort(scores)[-top_k:]
    argmax_scores = np.argsort(scores)[-top_k:]
    return file_name, max_scores, argmax_scores


def mapReduce(file_name, input_emb, top_k=5):
    input_path, csv_embeddings = mapper_file_to_embeddings(file_name)
    file_name, scores = mapper_embeddings_to_scores(input_path, csv_embeddings, input_emb)
    return reducer_scores_to_max(file_name, scores, top_k=top_k)


def thread_mapReduce(file_names, input_emb, top_k=5):
    futures = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for file_name in file_names:
            futures.append(executor.submit(mapReduce, file_name, input_emb, top_k=top_k))

    futures, _ = concurrent.futures.wait(futures)
    results = [future.result() for future in futures]

    df = pd.DataFrame(columns=file_names)

    return results


def main():
    emb = np.load("sunflower/sunflower1.npy")
    file_names = [line.strip() for line in open("file_names.txt", "r")]
    file_names = file_names[:10]
    results = thread_mapReduce(file_names, emb)
    for csv_result in results:
        csv_name = csv_result[0][0]
        print(f"Results for {csv_name}")
        for result in csv_result:
            print(result)


def save_names():
    path = "images"
    file_names = glob.glob(os.path.join(path, "*.hdf5"))
    # remove the path
    file_names = [os.path.basename(file_name).replace(".hdf5", "") for file_name in file_names]

    # save the file names to a text file
    with open("file_names.txt", "w") as f:
        for file_name in file_names:
            f.write(file_name + "\n")


if __name__ == '__main__':
    # save_names()
    main()