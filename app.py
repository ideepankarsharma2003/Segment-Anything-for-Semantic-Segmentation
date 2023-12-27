import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

from utils.sam_predictor import segment_image

drawing_mode= "rect"
stroke_width= 1
stroke_color= "purple"
bg_color= "white"
bg_image= st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=480,
    width=720,
    drawing_mode=drawing_mode,
    point_display_radius= 0,
    key="canvas",
)

if canvas_result.json_data is not None:
    # print(canvas_result.json_data)
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    if "left" in objects.columns:    
        objects= objects.rename(columns={
            "left":"x",
            "top":"y",
            
        })
        st.dataframe(objects[["x", "y", "width", "height"]])
    

   
if bg_image:
    image= Image.open(bg_image)
    
    image= image.resize((720,480))
    x= st.number_input(label="x", value=0)
    y= st.number_input(label="y", value=0)
    width= st.number_input(label="width", value=image.width)
    height= st.number_input(label="height", value=image.height)
    segment= st.button("segment")
    if segment:
        print(image.width,image.height)
        print(x,y,width,height)
        st.image(image)
        
        box= np.array([
            x,
            y,
            x+width,
            y+height
        ])
        
        image= segment_image(pil_image=image, box=box)
        st.image(image)
    
    
    
