import time

import streamlit as st

SAVE_PATH = "uploads/"


def init_model():
    """
    Initialize the model.
    """
    # Load the model
    from imager.model import AmazonModel
    predictor = AmazonModel()
    return predictor


def save_image_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    file_name = SAVE_PATH + "image." + file_extension
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name


def predict_link(model, image_path):
    """
    Predict the link of the image.

    :param model: The model to use for prediction
    :param image_path: The path of the image to predict
    :return: The predicted link
    """
    image_bytes = open(image_path, "rb").read()
    return model.predict(image_bytes)


def stream_data(text):
    """
    Stream the text data.

    :param text: The text to stream
    """
    for word in text:
        yield word
        time.sleep(0.02)

def main():
    st.title("Amazon Image Predictor 🖼️")

    model = init_model()

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file:
        file_extension = "png"
        image_bytes = image_file.read()
        # show image
        st.image(image_bytes, caption='Uploaded Image', use_column_width=True)
        save_image_file(image_bytes, file_extension)


    # Display the transcript
    st.header("Predictions")

    if st.button("Predict"):
        # Transcribe the audio file
        links = predict_link(model, SAVE_PATH + "image.png")

        # Bullet points of link
        for link in links:
            name = link.split("/")[3]
            st.markdown(f"- [{name}]({link})")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            padding-left:40px;
        }
        </style>
        ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()