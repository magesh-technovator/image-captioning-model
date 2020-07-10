#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:07:19 2020

@author: MAGESHWARAN
"""

import streamlit as st
import yaml
from build_vocab import Vocabulary
from inference import Predictor

# Load model configuration
with open("config.yaml", "rb") as rb:
    config = yaml.load(rb, yaml.FullLoader)
    
model = Predictor(config)

# UI Components
st.title("Image Captioning Model")

st.write("Upload your image and Generate Caption automatically")

image = st.file_uploader("Insert Image")


if st.button("Generate Caption"):
    if image is None:
        st.write("Insert image to generate Caption")
        
    else:
        prediction = model.predict(image)
        prediction = prediction.strip(".").strip()
        st.write(prediction)
