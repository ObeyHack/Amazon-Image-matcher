import sklearn
from sklearn.cluster import KMeans
from imager.image_classifier import get_embedding_links
import pandas as pd
import matplotlib.pyplot as plt


def main():
    k = 139
    embs = get_embedding_links(csv_folder_path=r'..\Datasets')
    embs = embs[0]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embs)
    predictions = kmeans.predict(embs)

    # plot the clusters using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    embs_pca = pca.fit_transform(embs)

    # plot the clusters using PCA, 2D
    fig, ax = plt.subplots()
    for i in range(k):
        # label is equal to the degree of the most common label in the cluster
        label = pd.Series(predictions[predictions == i]).mode()[0]
        ax.scatter(embs_pca[predictions == i, 0], embs_pca[predictions == i, 1], label=label)
    # ax.legend()  # TODO: fix legend if needed
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('2D PCA plot of image clusters')
    plt.show()

    # plot the clusters using PCA, 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(k):
        # label is equal to the degree of the most common label in the cluster
        label = pd.Series(predictions[predictions == i]).mode()[0]
        ax.scatter(embs_pca[predictions == i, 0], embs_pca[predictions == i, 1], embs_pca[predictions == i, 2],
                   label=label)
    # ax.legend()  #TODO: fix legend if needed
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('3D PCA plot of image clusters')
    plt.show()

    # plot histogram of the clusters
    plt.hist(predictions, bins=k)
    plt.xlabel('Cluster')
    plt.ylabel('Number of images')
    plt.title('Histogram of image clusters')
    plt.show()

    # # df = pd.read_csv(r'..\Datasets\Amazon-Products.csv')
    # # rating = df['ratings']
    # # # make nan zero
    # # rating = rating.fillna(0)
    # # # change 'Get' to 0
    # # rating = rating.replace('Get', 0)
    # # rating = rating.replace('FREE', 0)
    # # for i in range(len(rating)):
    # #     # if rating[i] starts with ₹, change to 0
    # #     # if
    # #     if rating[i] == 0:
    # #         continue
    # #     if rating[i].startswith('₹'):
    # #         rating[i] = 0
    # # 
    # # # make float from string
    # # rating = rating.str.replace(',', '').astype(float)
    # # # change rows with ₹ to 0
    # # 
    # # price = df['actual_price']
    # # # make float from string
    # # price = price.str.replace('₹', '').str.replace(',', '').astype(float)
    # # price = price.fillna(0)
    # # 
    # # # plot scatter plot of clusters
    # # plt.scatter(rating, price, c=predictions)
    # # plt.xlabel('Rating')
    # # plt.ylabel('Price')
    # # plt.title('Scatter plot of image clusters')
    # # plt.show()


if __name__ == '__main__':
    main()
