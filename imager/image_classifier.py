import glob
import os
import numpy as np
import h5py
import requests
from keras.src.losses import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn

from imager.vgg.VGG import load_image_bytes, get_image_embeddings


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
    return results


def get_embedding_links(return_dict=False, hdf5_folder_path="images", csv_folder_path=os.getcwd() + r"\Datasets",
                        return_subjects=False):
    # Initialize an empty list to store embeddings and image links
    all_embeddings = []
    image_links = []
    all_subjects = []

    def _list_datasets(file_path):
        """List all dataset names in an HDF5 file."""
        with h5py.File(file_path, 'r') as hdf:
            return [name for name in hdf.keys()]

    # Iterate through each file in the folder
    for file_name in os.listdir(hdf5_folder_path):
        # if the hdf5 file is dataset.hdf5, continue
        if file_name.endswith('.hdf5'):
            file_path = os.path.join(hdf5_folder_path, file_name)
            csv_file_path = os.path.join(csv_folder_path, file_name.replace('.hdf5', '.csv'))

            dataset_names = _list_datasets(file_path)

            with h5py.File(file_path, 'r') as hdf:
                # Read the corresponding CSV file to get image links
                df = pd.read_csv(csv_file_path)
                for dataset_name in dataset_names:
                    embeddings = hdf[dataset_name][:]
                    if embeddings.size > 0:
                        # Detect the actual embedding size
                        actual_embedding_size = embeddings.shape[-1]
                        for i, embedding in enumerate(embeddings):
                            all_embeddings.append(embedding)
                            # append the "subject" column value to the all_subjects list
                            all_subjects.append(df['main_category'].iloc[i])
                            image_links.append(df['image'].iloc[i])
                    else:
                        all_embeddings.append(np.zeros(actual_embedding_size))  # Handle empty embeddings case
                        all_subjects.append("Empty category")  # Handle empty embeddings case
                        image_links.append(None)  # Handle empty embeddings case

    # Convert lists to NumPy arrays
    all_embeddings = np.array(all_embeddings)
    image_links = np.array(image_links)
    all_subjects = np.array(all_subjects)

    # Ensure the embeddings array is 2-dimensional
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    if return_dict:
        # create a dictionary with embeddings to image links
        embedding_links_dict = {}
        for i in range(len(all_embeddings)):
            embedding_links_dict[tuple(all_embeddings[i])] = image_links[i]
        return embedding_links_dict

    if return_subjects:
        return all_embeddings, all_subjects

    return all_embeddings, image_links


def get_cosine_similarities(query_embedding, all_embeddings, k=0):
    # Calculate cosine similarities
    similarities = cosine_similarity_sklearn(query_embedding.reshape(1, -1), all_embeddings).flatten()
    if k > 0:
        if k > len(similarities):
            print(f"Warning: k is larger than the number of embeddings. Setting k to {len(similarities)}")
            k = len(similarities)

        # Get indices of top k closest embeddings
        top_k_indices = np.argsort(-similarities)[:k]
        return similarities, top_k_indices
    return similarities


def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    img_data = response.content
    return img_data


def embed_image_from_url(url):
    img_data = download_image(url)
    if img_data is None:
        return np.zeros((1, 512))
    img = load_image_bytes(img_data)
    embedding = get_image_embeddings(img)
    return embedding


def main():
    emb = np.load("modeler/sunflower/sunflower1.npy")
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
