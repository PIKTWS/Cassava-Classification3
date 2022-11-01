# Install library
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
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

# Process-Classification
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
        temp_preds = preds
        preds = np.array(preds[0]).tolist()
        for i in range(0,len(preds)):
                preds[i] = preds[i]*100
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


# Result
        st.write("Cassava Bacterial Blight (CBB) [โรคใบไหม้]: ",preds[0])
        st.write("Cassava Brown Streak Disease (CBSD) [โรคใบจุดสีน้ำตาล]: ",preds[1])
        st.write("Cassava Green Mottle (CGM) [ติดเชื้อไวรัสมอสสีเขียว]: ",preds[2])
        st.write("Cassava Mosaic Disease (CMD) [โรคใบด่าง]: ",preds[3])
        st.write("Healthy: ",preds[4])
        
        #chart_data = pd.DataFrame(data = {"CBB":round(preds[0],2),"CBSD":round(preds[1],2),"CGM":round(preds[2],2),"CMD":round(preds[3],2),"Healthy":round(preds[4],2)}, index=[0])
            

        #st.bar_chart(chart_data)
        if result == "Healthy":
                st.balloons()
                st.write("")
                st.success('Good News! This leaf is healthy')
        else:
            st.error("Oh no! this leaf is "+result)
            st.info("You Should ........")

        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer('top_conv')
            iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(img)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

        heatmap = heatmap[0, :, :]
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = image
        INTENSITY = 0.5

        heatmap = cv2.resize(heatmap, (img.width, img.height))

        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        img = heatmap * INTENSITY + img

        st.image(img,use_column_width=True)
