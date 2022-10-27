import sys 

sys.path.append('/usr/local/lib/python3.9/site-packages')

import streamlit as st
import pandas as pd
from pyecharts.charts import Pie
from numpy import array
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from colorblind import colorblind
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

st.set_page_config( page_title="Color Analysis with David",page_icon="🧊",layout="wide", initial_sidebar_state="expanded")      

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(//cdn.shopify.com/s/files/1/0011/4651/9637/t/220/assets/moonbackground.png?v=7157594…);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()   

streamlit_style = """
			<style>
			@import url("https://fonts.shopifycdn.com/nunito_sans/nunitosans_n7.5bd4fb9346d13afb61b3d78f8a1e9f31b128b3d9.woff2?h1=c3RvcmUudGF5bG9yc3dpZnQuY29t&hmac=60eccdbd2670eec353d0081774bf120b82d525923d9ffdd1d1f37aaf86d8bcb8") format("woff2"), url("https://fonts.shopifycdn.com/nunito_sans/nunitosans_n7.2bcf0f11aa6af91c784a857ef004bcca8c2d324d.woff?h1=c3RvcmUudGF5bG9yc3dpZnQuY29t&hmac=2bc2b91b35c84d85452727bfe0d6cc4ac43bf8978e2a27cfb29c154128a4a4cc") format("woff");

			html, body, [class*="css"]  {
			font-family: 'Nunito Sans', sans-serif;
			font-size: 50px;
			color: 	#a1bbc5;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Taylor Swift Midnights Album Vinyl Varients Color Analysis with David </p>', unsafe_allow_html=True)

#Read the images and Convert the color to RBG
#Read the images using imread method by OpenCV to read the image. And then, we are converting the color format from BGR to RGB using cvtColor
image1 = cv2.imread('Moon Stone Blue Vinyl.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.imshow(image1)

#Convert RGB Color to Hex Color
#In this function, we are converting an RGB color into Hex color format. This function will help at the end when visualizing the results of our analysis. Instead of having three different values (red, green, blue), we will have one output: hex value.

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    names= [convert_rgb_to_names(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize = (11.25, 11.25), edgecolor='Black')
    plt.pie(counts.values(), colors = hex_colors,labels=names,autopct='%11.2f%%')
    
    plt.savefig("color_analysis_report.png")
    print(hex_colors)
    
#Image Color Analyser
modified_image = prep_image(image1)
color_analysis(modified_image)


st.image("Midnights-Logo.png", width=200)
st.image("TS-Midnights-Logo.PNG", width=200)
#st.image("Moon Stone Blue Vinyl.png", caption="Original Photo", width=240, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
#st.image("color_analysis_report.png", caption="Color Analysis Pie Chart", width=1080, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
col1, col2= st.columns((1,1))

with st.container():
    with col1:
        st.image("Moon Stone Blue Vinyl.png", caption="Original Photo", width=960, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
        

with st.container():
    with col2:
        st.image("color_analysis_report.png", caption="Color Analysis Pie Chart", width=960, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
