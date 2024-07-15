import numpy as np
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from sklearn.metrics.pairwise import cosine_similarity
import io


vgg16 = VGG16(weights='imagenet', include_top=False,
              pooling='max', input_shape=(224, 224, 3))


def load_image(image_path):
    """
        -----------------------------------------------------
        Process the image provided.
        - Resize the image
        -----------------------------------------------------
        return resized image
    """

    input_image = Image.open(image_path)
    input_image = input_image.convert('RGB')
    resized_image = input_image.resize((224, 224))

    return resized_image


def load_image_bytes(image_bytes):
    """
        -----------------------------------------------------
        Process the image provided.
        - Resize the image
        -----------------------------------------------------
        return resized image
    """

    input_image = Image.open(io.BytesIO(image_bytes))
    input_image = input_image.convert('RGB')
    resized_image = input_image.resize((224, 224))
    return resized_image


def get_image_embeddings(embedding, verbose=0):
    """
      -----------------------------------------------------
      convert image into 3d array and add additional dimension for model input
      -----------------------------------------------------
      return embeddings of the given image
    """
    #
    # image_array = np.expand_dims(object_image, axis=0)
    #

    image = img_to_array(embedding)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image_embedding = vgg16.predict(image, verbose=verbose)
    return image_embedding


def get_score(embedding1, embedding2):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image

    """
    similarity_score = cosine_similarity(embedding1, embedding2)
    return similarity_score


def get_similarity_score(first_image: str, second_image: str):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image

    """
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = get_score(first_image_vector, second_image_vector)
    return similarity_score


def image2label(image):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image

    """
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image_embedding = vgg16.predict(image)
    label = decode_predictions(image_embedding)
    return label


def label(embedding):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image

    """
    label = decode_predictions(embedding)
    return label


if __name__ == '__main__':
    img = load_image('sunflower/sunflower1.jpg')
    img_embedding = get_image_embeddings(img)
    # score(img_embedding, img_embedding)
    # save the embeddings
    np.save('sunflower/sunflower1.npy', img_embedding)
