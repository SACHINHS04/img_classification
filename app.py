import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

# load the pre-trained model
model = vgg16.VGG16(weights='imagenet')

def classify_image(img):
    img = img.resize((224, 224)) # Resize the image
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    preds = model.predict(img)
    return vgg16.decode_predictions(preds, top=3)[0]

def main():
    st.header("Image Classification App")
    st.text("Upload an image and the model will predict what's in it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # convert the uploaded file to a PIL image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        predictions = classify_image(image)
        for pred in predictions:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
        # clear the cache after the classification is done
        st.cache.clear()

if __name__ == '__main__':
    main()
