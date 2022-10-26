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

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.set_page_config( page_title="Color Analysis with David",page_icon="🧊",layout="wide", initial_sidebar_state="expanded")
st.subheader('Color Analysis with David')

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
col1, col2= st.columns((2,1))

#local_css("style.css")

with st.container():
    with col1:
        st.image("Moon Stone Blue Vinyl.png", caption="Original Photo", width=720, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#local_css("style.css")

with st.container():
    with col2:
        st.image("color_analysis_report.png", caption="Color Analysis Pie Chart", width=1080, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
