# Install library
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import streamlit as st

# load model
model = tf.keras.models.load_model("./EffNetB0_model.h5")

# Make up page
st.image("./pics/cropped-logo.png", caption=None, use_column_width=True, clamp=False, channels='RGB',output_format='auto')
st.title('🌿Cassava Classification🍃')


# Upload file
file = st.file_uploader("Please upload Cassava Leaf", type=["jpg"])
if file is not None:
        st.image(file,use_column_width=True)
# Function
def test_model(file,img_shape=512):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    """
        # Resize the image
    img = tf.image.decode_jpeg(file)
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    #** Change size & dimention image

    # img = tf.expand_dims(img, axis=0)
    # x = image.img_to_array(file)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    preds = model.predict(img)
    preds = np.array(preds).tolist()
    temp = preds.index(max(preds))
    if temp == 0 :
        return "Cassava Bacterial Blight (CBB) [โรคใบไหม้]"
    elif temp == 1 :
        return "Cassava Brown Streak Disease (CBSD) [โรคใบจุดสีน้ำตาล]"
    elif temp == 2 :
        return "Cassava Green Mottle (CGM) [ติดเชื้อไวรัสมอสสีเขียว]"
    elif temp == 3 :
        return "Cassava Mosaic Disease (CMD) [โรคใบด่าง]"
    elif temp == 4 :
        return "Healthy"

    """
    # Read in the image
    img = tf.io.read_file(file)
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
    """
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
