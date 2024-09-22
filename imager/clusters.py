import sklearn
from sklearn.cluster import KMeans
from imager.image_classifier import get_embedding_links


def main():
    k=5
    embs = get_embedding_links()
    embs = embs[0]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embs)
    predictions = kmeans.predict(embs)
    print(predictions)
    # plot histogram of the clusters
    import matplotlib.pyplot as plt
    plt.hist(predictions, bins=k)
    plt.xlabel('Cluster')
    plt.ylabel('Number of images')
    plt.title('Histogram of image clusters')
    plt.show()
    # read C:\Users\merzi\PycharmProjects\pythonProject\Amazon-Image-matcher\imager\Datasets\Amazon-Products.csv
    import pandas as pd
    df = pd.read_csv('Datasets\Amazon-Products.csv')
    rating = df['ratings']
    #make nan zero
    rating = rating.fillna(0)
    #change 'Get' to 0
    rating = rating.replace('Get',0)
    rating = rating.replace('FREE', 0)
    for i in range(len(rating)):
        #if rating[i] starts with ₹, change to 0
        #if
        if rating[i]==0:
            continue
        if rating[i].startswith('₹'):
            rating[i] = 0

    #make float from string
    rating = rating.str.replace(',','').astype(float)
    #change rows with ₹ to 0


    price = df['actual_price']
    #make float from string
    price = price.str.replace('₹','').str.replace(',','').astype(float)
    price = price.fillna(0)

    # plot scatter plot of clusters
    plt.scatter(rating, price, c=predictions)
    plt.xlabel('Rating')
    plt.ylabel('Price')
    plt.title('Scatter plot of image clusters')
    plt.show()



if __name__ == '__main__':
    main()
