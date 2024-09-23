import sklearn
from sklearn.cluster import KMeans
from imager.image_classifier import get_embedding_links
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tqdm


def main():
    link_embeddings, subjects = get_embedding_links(csv_folder_path=r'..\Datasets', return_subjects=True)

    # plot the link embeddings according to their corresponding subjects, using 2D PCA and legend
    pca = PCA(n_components=2)
    embs_pca = pca.fit_transform(link_embeddings)
    fig, ax = plt.subplots()

    unique_subjects = pd.Series(subjects).unique()
    num_of_subjects = len(unique_subjects)

    # create a distinct color map for the subjects, where the colors are very different from each other
    colors = plt.cm.get_cmap('tab20', num_of_subjects)
    colors = colors(range(num_of_subjects))
    # color_map = plt.cm.get_cmap('hsv', num_of_subjects)
    # colors = [color_map(i) for i in range(num_of_subjects)]

    # for each unique subject, plot the corresponding embeddings that belong to that subject
    for subject in tqdm.tqdm(unique_subjects):
        ax.scatter(embs_pca[subjects == subject, 0], embs_pca[subjects == subject, 1], label=subject,
                   c=colors[unique_subjects.tolist().index(subject)], alpha=0.7, s=10)
    ax.legend()
    # make sure the legend is not cut off, and that it is not on the graph
    # plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('2D PCA plot of image embeddings')
    plt.tight_layout()
    plt.show()
    fig.savefig(r'plots\2D_PCA_plot.png')

    # plot the link embeddings according to their corresponding subjects, using 3D PCA and legend
    pca = PCA(n_components=3)
    embs_pca = pca.fit_transform(link_embeddings)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for each unique subject, plot the corresponding embeddings that belong to that subject
    for subject in tqdm.tqdm(unique_subjects):
        ax.scatter(embs_pca[subjects == subject, 0], embs_pca[subjects == subject, 1], embs_pca[subjects == subject, 2],
                   label=subject, c=colors[unique_subjects.tolist().index(subject)], alpha=0.7, s=10)
    ax.legend()
    # make sure the legend is not cut off, and that it is not on the graph
    # make sure each subject has a distinctive color
    # plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('3D PCA plot of image embeddings')
    plt.tight_layout()
    plt.show()

    # plot the link embeddings according to their corresponding subjects, using 3D PCA and legend - and save 8 images
    # where each image is a different angle of the 3D plot (0, 45, 90, 135, 180, 225, 270, 315 degrees)
    for angle in range(0, 360, 45):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, angle)
        # for each unique subject, plot the corresponding embeddings that belong to that subject
        for subject in tqdm.tqdm(unique_subjects):
            ax.scatter(embs_pca[subjects == subject, 0], embs_pca[subjects == subject, 1],
                       embs_pca[subjects == subject, 2], label=subject,
                       c=colors[unique_subjects.tolist().index(subject)], alpha=0.7, s=10)
        ax.legend()
        # make sure the legend is not cut off, and that it is not on the graph
        # make sure each subject has a distinctive color
        # plt.subplots_adjust(right=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        ax.set_title('3D PCA plot of image embeddings')
        plt.tight_layout()
        plt.show()
        # save the plot
        fig.savefig(fr'plots\3D_PCA_plot_{angle}degrees.png')

    # # Clusters part
    # k = len(pd.Series(subjects).unique())
    # print(k, pd.Series(subjects).unique())
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(link_embeddings)
    # predictions = kmeans.predict(link_embeddings)
    # 
    # # plot the clusters using PCA
    # pca = PCA(n_components=3)
    # embs_pca = pca.fit_transform(link_embeddings)
    # 
    # # plot the clusters using PCA, 2D
    # fig, ax = plt.subplots()
    # # # label according to subject
    # # for subject in tqdm.tqdm(pd.Series(subjects).unique()):
    # #     ax.scatter(embs_pca[subjects == subject, 0], embs_pca[subjects == subject, 1], label=subject)
    # # ax.legend()
    # # label according to cluster
    # for i in range(k):
    #     # label is equal to the degree of the most common label in the cluster
    #     label = pd.Series(predictions[predictions == i]).mode()[0]
    #     ax.scatter(embs_pca[predictions == i, 0], embs_pca[predictions == i, 1], label=label)
    # plt.subplots_adjust(right=0.7)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_title('2D PCA plot of image clusters')
    # plt.show()

    # # plot the clusters using PCA, 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(k):
    #     # label is equal to the degree of the most common label in the cluster
    #     label = pd.Series(predictions[predictions == i]).mode()[0]
    #     ax.scatter(embs_pca[predictions == i, 0], embs_pca[predictions == i, 1], embs_pca[predictions == i, 2],
    #                label=label)
    # ax.legend()  # TODO: fix legend if needed
    # # for subject in tqdm.tqdm(pd.Series(subjects).unique()):
    # #     ax.scatter(embs_pca[subjects == subject, 0], embs_pca[subjects == subject, 1], embs_pca[subjects == subject, 2],
    # #                label=subject)
    # plt.subplots_adjust(right=0.7)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_zlabel('PCA 3')
    # ax.set_title('3D PCA plot of image clusters')
    # plt.show()
    # 
    # # plot histogram of the clusters
    # plt.hist(predictions, bins=k)
    # plt.xlabel('Cluster')
    # plt.ylabel('Number of images')
    # plt.title('Histogram of image clusters')
    # plt.show()

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
