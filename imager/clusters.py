import sklearn
from sklearn.cluster import KMeans
from imager.image_classifier import get_embedding_links
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tqdm


def plot_2D_PCA(link_embeddings, subjects, unique_subjects, colors, embs_pca):
    # plot the link embeddings according to their corresponding subjects, using 2D PCA and legend
    fig, ax = plt.subplots()

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


def plot_3D_PCA(link_embeddings, subjects, unique_subjects, colors, embs_pca):
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


def find_best_k_clusters(link_embeddings):
    # plot the graph as scatter plot of k vs inertia
    k_values = range(2, 139)
    inertias = []
    for k in tqdm.tqdm(k_values):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(link_embeddings)
        inertias.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(k_values, inertias)
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia vs k')
    plt.tight_layout()
    plt.show()
    fig.savefig(r'plots\Inertia_vs_k.png')

    # return best k
    return inertias.index(min(inertias)) + 2


def plot_clusters(link_embeddings, k, embs_pca):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(link_embeddings)
    fig, ax = plt.subplots()
    ax.scatter(embs_pca[:, 0], embs_pca[:, 1], c=kmeans.labels_, s=10, cmap='tab20')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(f'{k} clusters of image embeddings')
    plt.tight_layout()
    plt.show()
    fig.savefig(fr'plots\{k}_clusters.png')


def main():
    link_embeddings, subjects = get_embedding_links(csv_folder_path=r'..\Datasets', return_subjects=True)

    unique_subjects = pd.Series(subjects).unique()
    num_of_subjects = len(unique_subjects)

    # create a distinct color map for the subjects, where the colors are very different from each other
    colors = plt.cm.get_cmap('tab20', num_of_subjects)
    colors = colors(range(num_of_subjects))

    pca = PCA(n_components=3)
    embs_pca = pca.fit_transform(link_embeddings)

    # # 2D PCA part
    # plot_2D_PCA(link_embeddings, subjects, unique_subjects, colors, embs_pca)
    # 
    # # 3D PCA part
    # plot_3D_PCA(link_embeddings, subjects, unique_subjects, colors, embs_pca)

    # Clusters part (Not working well)
    # find the best k clusters
    k = find_best_k_clusters(link_embeddings)

    # plot the best k clusters.
    plot_clusters(link_embeddings, k, embs_pca)


if __name__ == '__main__':
    main()
