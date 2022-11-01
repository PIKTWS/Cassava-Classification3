# Install library
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# load model
model = tf.keras.models.load_model("./EffNetB0_model.h5")

# Make up page
st.image("./pics/cropped-logo.png", caption=None, width=200, use_column_width=None, clamp=False, channels='RGB',output_format='auto')
st.title('🌿Cassava Classification🍃')


# Upload file
file = st.file_uploader("Please upload Cassava Leaf", type=["jpg"])
# Function
def test_model(file,img_shape=512):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    """
    # Read in the image
    img = Image.open(file)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    final = "no result"
    #** Change size & dimention image
    img = tf.expand_dims(img, axis=0)
    resultOfModel = model.predict(img)
    result = resultOfModel.argmax() 
    if result == 0:
        final = "Cassava Bacterial Blight"
    elif result == 1:
        final = "Cassava Brown Streak Disease"
    elif result == 3:
        final = "Cassava Mosaic Disease"
    elif result == 4:
        final = "Healthy"
    else :
        final = "ระบุไม่ได้"
    return final
# Process-Classification

submit = st.button('Predict')
if submit:
    if file is None:
        st.text("Please upload an image file")
    else :      
        result = test_model(file)
        if result == "Healthy":
                st.balloons()
                st.write("")
                st.success('Good News! This leaf is healthy')
        else:
            st.error("Oh no! this leaf is "+result)
            st.info("You Should ........")


# Result
