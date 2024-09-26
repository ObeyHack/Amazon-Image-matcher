import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from imager import text_evaluation
from imager.image_classifier import get_embedding_links, get_cosine_similarities
from imager.vgg.VGG import get_image_embeddings, load_image_bytes




class AmazonModel:
    def __init__(self, root_path=os.getcwd(),k=5):
        all_embeddings, image_links = get_embedding_links(hdf5_folder_path=root_path + "/imager/images/")
        self.k = k
        self.embeddings = all_embeddings
        self.root_path = root_path
        self.data = pd.read_csv(root_path + "/imager/Datasets/Amazon-Products.csv")

    def predict(self, x):
        """
        :param x: image bytes
        :return: array of links (str) - the top k predictions
        """
        img = load_image_bytes(x)
        query_emb = get_image_embeddings(img)
        similarities, top_indices = get_cosine_similarities(query_emb, self.embeddings, k=self.k)
        links = []
        for i in top_indices:
            links.append(self.data.iloc[i]['link'])

        valid_links = []
        for link in links:
            try:
                text_evaluation.get_soup_retry(link)

            except AssertionError:
                continue
            valid_links.append(link)

        return valid_links

    def eval(self, X, Y):
        """
        Evaluate the model using the test data, the test data will be (image,text) pairs.
        :param X: batch of images
        :param Y: batch of links to the text descriptions
        :return: average evaluation score
        """
        def un_batched_eval(self, x, y):
            """
            :param x: an image
            :param y: a link to the text description
            :return: int - evaluation score
            """
            top_k_predictions = self.predict(x)
            y_emb = text_evaluation.text_embedding(text_evaluation.description_scraper(y))
            predictions_emb = [text_evaluation.text_embedding(text_evaluation.description_scraper(link))
                               for link in top_k_predictions]

            similarity_scores = [text_evaluation.text_score(y_emb, emb) for emb in predictions_emb]
            return np.mean(similarity_scores), np.std(similarity_scores)

        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")

        scores = [un_batched_eval(self, X[i], Y[i]) for i in tqdm(range(len(X)))]
        return scores


