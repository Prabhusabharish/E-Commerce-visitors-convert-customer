import numpy as np
import streamlit as st
# from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import joblib
import keras_ocr
import easyocr
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import subprocess
import mysql.connector
import cv2
import re
import pytesseract
import folium
from folium.plugins import MarkerCluster
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
#Recommendations
import requests
import base64
from flask import Flask, request, jsonify
#--------------------------------------------------------------addbar-------------------------
icon = Image.open("1.png")
st.set_page_config(page_title= "E-Com",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This E-Com page is created by *Prabakaran!"""})
st.markdown("<h2 style='text-align: center; color: black;'>Predictive Analytics for E-Commerce: Converting Visitors into Valued Customers</h2>", unsafe_allow_html=True)

#--------------------------------------------------------------bgpage-------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Example usage
set_png_as_page_bg("2.jpg")


#--------------------------------------------------------------homepage-------------------------


def page1():
    # set_background()
    # Define columns
    col1, col2 = st.columns([2, 1])  # Making col1 wider and col2 narrower

    # Display image in col2
    with col2:
        st.image("Image/e-commerce-business-1024x683.jpg", use_column_width=True)
        st.image("Image/Ecommerce-Shopping-Infographics.png", use_column_width=True)
    # Display text content in col1
    with col1:
        st.markdown("""
        ## Project Overview

        - **DataFrame Analysis:** Explore detailed insights and outcomes from our ML models. Witness a comprehensive breakdown of model performances, including accuracy, precision, recall, and F1-score, neatly organized in a tabular format.
        - **Exploratory Data Analysis (EDA):** Embark on a visual journey through our dataset, unraveling hidden patterns and trends through interactive plots and charts.
        - **Prediction Engine:** Experience the power of predictive analytics as we forecast the likelihood of visitor conversion with real-time prediction probabilities.

        **Image Analysis:**
        - Utilize a range of image preprocessing techniques.
        - Extract and present text from images for further analysis.

        **NLP Text Preprocessing:**
        - Witness the transformation of raw text data through processes such as stemming, lowercasing, and more.
        - Explore sentiment analysis with vivid bar chart visualizations, providing invaluable insights into customer sentiments.
        - Immerse yourself in the art of word clouding to gain a holistic perspective of your textual data.

        **Recommendations System:**
        - Discover global recommendations crafted using custom movie datasets.
        - Personalized suggestions tailored to user preferences.
        """)


# #--------------------------------------------------------------page2-------------------------

def generate_model_table(model_name, accuracy, precision, recall, f1):
    model_table = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    return model_table

def page2():
    st.title("DataFrame")
    st.write("")

    path = "classification_data.csv"
    df = pd.read_csv(path)
    st.dataframe(df)

    # Placeholder values, replace with actual metrics
    mmodel_metrics = [
        ('Random Forest', 0.9932, 0.9955, 0.9918, 0.9936),
        ('Logistic Regression', 0.7546, 0.7313, 0.8549, 0.7883),
        ('Support Vector Machine', 0.7225, 0.7336, 0.7550, 0.7441),
        ('K-Nearest Neighbors', 0.9686, 0.9671, 0.9745, 0.9708),
        ('Decision Tree', 0.9898, 0.9914, 0.9896, 0.9905)
    ]

    # Create a list to store DataFrames for each model
    model_tables = []

    # Generate model tables and store in the list
    for model_name, accuracy, precision, recall, support, f1 in model_metrics:
        model_table = generate_model_table(model_name, accuracy, precision, recall, support, f1)
        model_tables.append(model_table)

    # Concatenate DataFrames for all models
    df_results = pd.concat(model_tables, ignore_index=True)

    # Display the DataFrame using Streamlit
    st.title("Algorithm results")
    st.dataframe(df_results)

# #--------------------------------------------------------------page3-------------------------

def page3():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("")

    # Load the CSV file into a DataFrame
    path = "classification_data.csv"
    df = pd.read_csv(path)

    # Define columns
    col1, col2, col3 = st.columns(3)

    # Visualizations in col1
    with col1:
        # Histogram for 'count_session'
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.histplot(df['count_session'], bins=30, kde=True, color='blue', ax=ax1)
        plt.title('Distribution of count_session')
        plt.xlabel('count_session')
        plt.ylabel('Frequency')
        st.pyplot(fig1)

    # Visualizations in col2
    with col2:
        # Count plot for 'channelGrouping'
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='channelGrouping', data=df, palette='viridis', ax=ax2)
        plt.title('Count of channelGrouping')
        plt.xlabel('channelGrouping')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # Visualizations in col3
    with col3:
        # Bar chart for total visitors vs. buyed visitors
        total_visitors = df['has_converted'].count()
        buyed_visitors = df['has_converted'].sum()

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=['Total Visitors', 'Buyed Visitors'], y=[total_visitors, buyed_visitors], palette='viridis', ax=ax3)
        ax3.set_ylabel('Count')
        ax3.set_title('Total Visitors vs. Buyed Visitors')

        # Show the chart
        st.pyplot(fig3)

    # Visualizations in col2 (continued)
    with col2:
        # Scatter Plot
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='time_on_site', y='transactionRevenue', data=df, ax=ax4)
        plt.title('Scatter Plot: time_on_site vs. transactionRevenue')
        plt.xlabel('Time on Site')
        plt.ylabel('Transaction Revenue')
        st.pyplot(fig4)

        # Pair Plot
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        numerical_columns = ['count_session', 'count_hit', 'avg_session_time', 'time_on_site', 'transactionRevenue']
        sns.pairplot(df[numerical_columns], height=2, diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.suptitle('Pair Plot of Numerical Columns', y=1.02)
        st.pyplot(fig5)

        # Boxplots for numerical variables
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='time_on_site', data=df, ax=ax6)
        plt.title('Boxplot: time_on_site')
        st.pyplot(fig6)


# #--------------------------------------------------------------page4-------------------------
bin_mapping = {
    'avg_session_time_binned': {'low': [0, 500], 'medium': [500, 1000], 'high': [1000, np.inf]},
    'count_hit_binned': {'low': [0, 100], 'medium': [100, 200], 'high': [200, np.inf]},
    'num_interactions_binned': {'low': [0, 1000], 'medium': [1000, 5000], 'high': [5000, np.inf]}
}

avg_session_time_options = list(bin_mapping['avg_session_time_binned'].keys())
count_hit_options = list(bin_mapping['count_hit_binned'].keys())
num_interactions_options = list(bin_mapping['num_interactions_binned'].keys())

df1 = pd.read_csv('df1.csv')  

sessionQualityDim_options = df1['sessionQualityDim'].unique().tolist()
single_page_rate_options = df1['single_page_rate'].unique().tolist()

def predict(model, features):
    df = pd.DataFrame(features, columns=['sessionQualityDim', 'avg_session_time_binned', 'single_page_rate', 'count_hit_binned', 'num_interactions_binned'])

    
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[:, 1]  
    return prediction, prediction_proba

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model('random_forest_model.pkl')

def page4():
    st.title("Prediction")

    selected_num_interactions = st.selectbox('Select num_interactions:', options=num_interactions_options)
    selected_avg_session_time = st.selectbox('Select avg_session_time:', options=avg_session_time_options)
    selected_sessionQualityDim = st.selectbox('Select sessionQualityDim:', options=sessionQualityDim_options)
    selected_count_hit = st.selectbox('Select count_hit:', options=count_hit_options)
    selected_single_page_rate = st.selectbox('Select single_page_rate:', options=single_page_rate_options)

    if st.button('Predict'):
        features = np.array([[selected_num_interactions, selected_avg_session_time, 
                              selected_sessionQualityDim, selected_count_hit, 
                              selected_single_page_rate]])

        feature_names = ['num_interactions', 'avg_session_time', 'sessionQualityDim', 'count_hit', 'single_page_rate']
        features_df = pd.DataFrame(features, columns=feature_names)

        prediction, prediction_proba = predict(model, features_df)
        
        with st.container():
            st.write('Prediction:', "Yes" if prediction == 1 else "No")
            st.write('Prediction Probability:', prediction_proba)

# #--------------------------------------------------------------page5-------------------------

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_resize(original_image, target_size=(200, 200)):
    resized_image = original_image.resize(target_size)
    return resized_image

def preprocess_contrast(original_image, factor=2.0):
    contrast_image = ImageEnhance.Contrast(original_image).enhance(factor)
    return contrast_image

def preprocess_brightness(original_image, factor=1.5):
    brightness_image = ImageEnhance.Brightness(original_image).enhance(factor)
    return brightness_image

def preprocess_rotation(original_image, rotation_angle=45):
    rotated_image = ImageOps.exif_transpose(original_image.rotate(rotation_angle))
    return rotated_image

def preprocess_flip(original_image, flip_type="horizontal"):
    if flip_type == "horizontal":
        flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_type == "vertical":
        flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        flipped_image = original_image
    return flipped_image

def preprocess_crop(original_image, coordinates=(50, 50, 150, 150)):
    cropped_image = original_image.crop(coordinates)
    return cropped_image

def preprocess_grayscale(original_image):
    grayscale_image = original_image.convert("L")
    return grayscale_image

def preprocess_edge_detection(original_image):
    img_array = np.array(original_image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    processed_image = ImageOps.grayscale(Image.fromarray(edges))
    return processed_image

def preprocess_text_extraction(original_image):
    img_array = np.array(original_image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

def preprocess_color_space_conversion(original_image, color_space="HSV"):
    if color_space == "HSV":
        converted_image = original_image.convert("HSV")
    else:
        converted_image = original_image.convert("RGB")

    if color_space == "HSV":
        converted_image = converted_image.convert("RGB")

    return converted_image

def preprocess_histogram_equalization(original_image):
    if original_image.mode == 'RGBA':
        original_image = original_image.convert('RGB')

    equalized_image = ImageOps.equalize(original_image)
    return equalized_image

def preprocess_image_filtering(original_image, filter_type="gaussian"):
    if filter_type == "gaussian":
        filtered_image = original_image.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_type == "median":
        filtered_image = original_image.filter(ImageFilter.MedianFilter(size=3))
    else:
        filtered_image = original_image
    return filtered_image

# def preprocess_text_extraction(original_image):
#     img_array = np.array(original_image)
#     gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#     extracted_text = pytesseract.image_to_string(gray_image)

#     if not extracted_text:
#         return "No text found."

#     return extracted_text



def page5():
    st.title("Image-Analysis")

    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpeg", "jpg"])

    if uploaded_image is not None:
        original_image = Image.open(uploaded_image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)

        processed_resized_image = preprocess_resize(original_image)
        processed_contrast_image = preprocess_contrast(original_image, factor=2.0)
        processed_brightness_image = preprocess_brightness(original_image, factor=1.5)
        processed_rotation_image = preprocess_rotation(original_image, rotation_angle=45)
        processed_flip_horizontal_image = preprocess_flip(original_image, flip_type="horizontal")
        processed_crop_image = preprocess_crop(original_image, coordinates=(50, 50, 150, 150))
        processed_grayscale_image = preprocess_grayscale(original_image)
        processed_edge_image = preprocess_edge_detection(original_image)
        processed_color_space_image = preprocess_color_space_conversion(original_image, color_space="HSV")
        processed_equalized_image = preprocess_histogram_equalization(original_image)
        processed_filtered_image = preprocess_image_filtering(original_image, filter_type="gaussian")


        with col1:
            st.image(processed_resized_image, caption="Resized Image", use_column_width=True)
            st.image(processed_crop_image, caption="Cropped Image", use_column_width=True)
            st.image(processed_grayscale_image, caption="Grayscale Conversion Result", use_column_width=True)

        with col2:
            st.image(processed_contrast_image, caption="Contrast Adjustment Result", use_column_width=True)
            st.image(processed_brightness_image, caption="Brightness Adjustment Result", use_column_width=True)
            st.image(processed_color_space_image, caption="Color Space Conversion Result", use_column_width=True)
            st.image(processed_equalized_image, caption="Histogram Equalization Result", use_column_width=True)

        with col3:
            st.image(processed_rotation_image, caption="Rotation Result", use_column_width=True)
            st.image(processed_flip_horizontal_image, caption="Horizontal Flip Result", use_column_width=True)
            st.image(processed_edge_image, caption="Edge Detection Result", use_column_width=True)
            st.image(processed_filtered_image, caption="Image Filtering Result", use_column_width=True)
        
        

        extracted_text = preprocess_text_extraction(original_image)

        st.write("Extracted Text:")
        st.write(extracted_text)
        

# #--------------------------------------------------------------page6-------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')




# NLP preprocessing functions
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def perform_stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in words]
    return ' '.join(stemmed_text)

def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_text)

def remove_punctuation_special_chars(text):
    return ''.join(char for char in text if char not in string.punctuation)

def remove_numbers(text):
    return ''.join(char for char in text if not char.isdigit())

def get_part_of_speech(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return pos_tags

def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)

    if sentiment_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, sentiment_score

def generate_sentiment_bar_chart(sentiment_score):
    fig, ax = plt.subplots()
    # Calculate y-values for the bar chart
    positive_value = max(0, sentiment_score['pos'])
    negative_value = max(0, -sentiment_score['neg'])
    neutral_value = max(0, sentiment_score['neu'])
    
    # Specify colors for each bar
    colors = ['green', 'red', 'blue']

    sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[positive_value, negative_value, neutral_value], palette=colors, ax=ax)
    st.pyplot(fig)

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot() 

def page6():
    st.title("NLP-Preprocessing")

    # Get user input
    user_text = st.text_area("Enter your text:", "Prabhusabharish Aspiring a Data Scientist...")

    if st.button("Process Text"):
        results = []

        processed_text = user_text
        results.append(('Original Text:', processed_text))

        processed_text = processed_text.lower()
        results.append(('Lowercasing:', processed_text))

        processed_text = remove_stopwords(processed_text)
        results.append(('Remove Stopwords:', processed_text))

        processed_text = perform_stemming(processed_text)
        results.append(('Stemming:', processed_text))

        processed_text = perform_lemmatization(processed_text)
        results.append(('Lemmatization:', processed_text))

        processed_text = remove_punctuation_special_chars(processed_text)
        results.append(('Remove Punctuation and Special Characters:', processed_text))

        processed_text = remove_numbers(processed_text)
        results.append(('Remove Numbers:', processed_text))

        processed_text = get_part_of_speech(processed_text)
        results.append(('Port of Speech:', processed_text))

        # Display results
        for step, result in results:
            st.subheader(step)
            st.write(result)

    st.title("Sentiment Analysis")

    if st.button("Perform Sentiment Analysis"):
        sentiment, sentiment_score = perform_sentiment_analysis(user_text)

        # Display results
        st.subheader("Sentiment:")
        st.write(sentiment)

        st.subheader("Sentiment Scores:")
        st.write(f"Positive: {sentiment_score['pos']:.2f}")
        st.write(f"Negative: {sentiment_score['neg']:.2f}")
        st.write(f"Neutral: {sentiment_score['neu']:.2f}")
        st.write(f"Compound: {sentiment_score['compound']:.2f}")

        # Generate and display the sentiment bar chart
        generate_sentiment_bar_chart(sentiment_score)


    def generate_word_cloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    if st.button("Generate Word Cloud"):
        generate_word_cloud(user_text) 

#--------------------------------------------------------------page7-------------------------
# from requests.packages.urllib3.util.retry import Retry


def page7():
    st.title("RECOMMENDATIONS")

    def recommend(movie):
        index = movies_data[movies_data['title'] == movie].index
        if not index.empty:
            index = index[0]
            distances = sorted(list(enumerate(similarity_data[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []
            for i in distances[1:6]:
                recommended_movie_names.append(movies_data.iloc[i[0]]['title'])
            return recommended_movie_names
        else:
            st.warning("Movie not found in the database.")
            return []

    movies_data = pd.DataFrame.from_dict(pickle.load(open('movies.pkl', 'rb')))
    similarity_data = pickle.load(open('similarity.pkl', 'rb'))

    st.header('Movie Recommender System')
    movie_list = movies_data['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        recommended_movie_names = recommend(selected_movie)
        st.write("Recommended Movies:")
        for movie_name in recommended_movie_names:
            st.write(f"- {movie_name}")

#--------------------------------------------------------------Maingpage-------------------------

# Streamlit app
def main():
    st.sidebar.title("")

    st.sidebar.image("p.jpg", width=100)
    st.sidebar.title("Prabakaran T (Data Science Enthusiast)")
    

    # Add links to the sidebar
    page_options = ["Home", "DataFrame", "EDA", "Prediction", "Image-Analysis", "NLP-Preprocessing", "RECOMMENDATIONS"]
    selected_page = st.sidebar.radio("", page_options)

    # Display content based on the selected page
    if selected_page == "Home":
        page1()
    elif selected_page == "DataFrame":
        page2()
    elif selected_page == "EDA":
        page3 ()
    elif selected_page == "Prediction":
        page4 ()
    elif selected_page == "Image-Analysis":
        page5 ()
    elif selected_page == "NLP-Preprocessing":
        page6 ()
    elif selected_page == "RECOMMENDATIONS":
        page7 ()

if __name__ == "__main__":
    main()
