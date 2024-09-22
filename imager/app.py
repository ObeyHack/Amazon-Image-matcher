import streamlit as st

SAVE_PATH = "modeler/images"


def init_model():
    """
    Initialize the model.
    """
    # Load the model
    from model import AmazonModel
    predictor = AmazonModel()
    return predictor


def save_image_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    file_name = SAVE_PATH + "audio." + file_extension
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name



def main():
    st.title("Hebrew Transcription Chatbot 🤖")

    model = init_model()

    audio_file = st.file_uploader("Upload Image", type=["png"])
    if audio_file:
        file_extension = "png"
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        save_image_file(audio_bytes, file_extension)


    # Display the transcript
    st.header("Transcript")

    # if st.button("Transcribe"):
    #     # Transcribe the audio file
    #     transcript_text = transcribe_audio(model, SAVE_PATH + "audio.mp3")
    #
    #     # Stream the transcript
    #     message = st.chat_message("assistant")
    #     # message.write(transcript_text)
    #     message.write_stream(stream_data(transcript_text))


if __name__ == "__main__":
    # Run the main function
    main()