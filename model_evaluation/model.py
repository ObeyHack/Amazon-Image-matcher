import os
from imager.image_classifier import get_embedding_links, get_cosine_similarities
import pandas as pd
from imager.vgg.VGG import get_image_embeddings, load_image_bytes
import text_evaluation
import numpy as np
class AmazonModel:
    def __init__(self, root_path=os.getcwd() + "/../",k=5):
        all_embeddings, image_links = get_embedding_links(hdf5_folder_path=root_path + "imager/images/")
        self.k = k
        self.embeddings = all_embeddings
        self.root_path = root_path
        self.data = pd.read_csv(root_path + "Datasets/Amazon-Products.csv")


    def predict(self, x):
        """
        :param x: image bytes
        :return: array of links (str) - the top k predictions
        """
        img = load_image_bytes(x)
        query_emb = get_image_embeddings(img)
        similarities, top_10_indices = get_cosine_similarities(query_emb, self.embeddings, k=self.k)
        links = []
        for i in top_10_indices:
            links.append(self.data.iloc[i]['link'])

        return links

    def eval_model(self,X, y):
        # X is the test data, y is the labels of the test data, model is the model that we will evaluate
        # data
        accuracy = 0
        for i in range(len(X)):
            x = X[i]
            y = y[i]
            score = eval(x, y)
            # if score > 0.5:
            #    accuracy += 1
            # or
            accuracy += score

        return accuracy / len(X)

    def eval(self,x, y):
        """
        :param x: an image
        :param y: text description
        :return: int - evaluation score
        """
        top_k_predictions = self.predict(x)
        top_k_predictions_text = [text_evaluation.description_scraper(url) for url in top_k_predictions]
        similarity_scores = [text_evaluation.similarity_score(text, y) for text in top_k_predictions_text]
        return np.mean(similarity_scores)



def main():
    model = AmazonModel()
    model.predict()


if __name__ == '__main__':
    main()