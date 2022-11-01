# Install library
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# load model
model = tf.keras.models.load_model("./EffNetB0_model.h5")

# Make up page
st.image("./pics/cropped-logo.png", caption=None, use_column_width=True, clamp=False, channels='RGB',output_format='auto')
st.title('🌿Cassava Classification🍃')


# Upload file
file = st.file_uploader("Please upload Cassava Leaf", type=["jpg"])
if file is not None:
        st.image(file,use_column_width=True)

submit = st.button('Predict')
if submit:
    if file is None:
        st.text("Please upload an image file")
    else :      
        image = Image.open(file)
            # Resize the image
        img_array = np.array(image)
        img = tf.image.resize(img_array, size=(512,512))
        img = tf.expand_dims(img, axis=0)

        preds = model.predict(img)
        preds = np.array(preds[0]).tolist()
        temp = preds.index(max(preds))
        result =""
        if temp == 0 :
            result = "Cassava Bacterial Blight (CBB) [โรคใบไหม้]"
        elif temp == 1 :
            result = "Cassava Brown Streak Disease (CBSD) [โรคใบจุดสีน้ำตาล]"
        elif temp == 2 :
            result = "Cassava Green Mottle (CGM) [ติดเชื้อไวรัสมอสสีเขียว]"
        elif temp == 3 :
            result = "Cassava Mosaic Disease (CMD) [โรคใบด่าง]"
        elif temp == 4 :
            result = "Healthy"    
# Process-Classification
        st.write("CBB: ",preds[0])
        st.write("CBSD: ",preds[1])
        st.write("CGM: ",preds[2])
        st.write("CMD: ",preds[3])
        st.write("Healthy: ",preds[4])
        
        if result == "Healthy":
                st.balloons()
                st.write("")
                st.success('Good News! This leaf is healthy')
        else:
            st.error("Oh no! this leaf is "+result)
            st.info("You Should ........")


# Result
