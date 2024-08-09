import os
from imager.image_classifier import get_embedding_links, get_cosine_similarities
import pandas as pd
from imager.vgg.VGG import get_image_embeddings, load_image_bytes


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


    def eval(self, x, y):
        """
        :param x: an image
        :param y: text description
        :return: int - evaluation score
        """
        y_hat = self.predict(x) # get the top k predictions
        pass



def main():
    model = AmazonModel()
    model.predict()


if __name__ == '__main__':
    main()