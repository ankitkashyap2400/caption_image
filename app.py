from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np

from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.models import Model

# Feature extractor model
base_model = InceptionV3(weights="imagenet")

feature_model = Model(
    base_model.input,
    base_model.layers[-2].output
)

def extract_features(uploaded_file):

    image = load_img(
        uploaded_file,
        target_size=(299, 299)
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(
        image
    )

    feature = feature_model.predict(
        image,
        verbose=0
    )

    return feature
model = load_model("caption_model.keras")
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image_feature, tokenizer, max_length):

    in_text = "startseq"

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = model.predict(
            [image_feature, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = idx_to_word(
            yhat,
            tokenizer
        )

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace(
        "startseq", ""
    ).replace(
        "endseq", ""
    ).strip()
import pickle


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)
import streamlit as st

st.title("AI Image Caption Generator")


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file)

    feature = extract_features(
        uploaded_file
    )

    caption = predict_caption(
        model,
        feature,
        tokenizer,
        max_length
    )

    st.write(caption)