"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles, load_data
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from eda.exploratory_data_analysis import data_overview, data_visualisation, description, image_magic, null_values_viz, common_attributes_viz, word_clouds_viz,pub_year_viz, movie_genre_viz, budget_viz
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

movies_df = load_data('./../unsupervised_movie_data/movies.csv')
ratings_df = load_data('resources/data/ratings.csv')
tags_df = load_data('resources/data/tags.csv')
train_df = load_data('./../unsupervised_movie_data/train.csv')
test_df = load_data('./../unsupervised_movie_data/test.csv')
links_df = load_data('./../unsupervised_movie_data/links.csv')
imdb_df = load_data('./../unsupervised_movie_data/imdb_data.csv')
genome_tags = load_data('./../unsupervised_movie_data/genome-tags.csv')
genome_score = load_data('./../unsupervised_movie_data/genome-scores.csv')
ss = load_data('./../unsupervised_movie_data/sample_submission.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Exploratory Data Analysis"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.image('imgs/undraw_netflix_q00o.png', use_column_width=True)
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
    
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown("""
            In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.  
            With this context, we as EDSA students have constructucted a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.  
            Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.  
            The evaluation metric for this model is Root Mean Square Error. Root Mean Square Error (RMSE) is commonly used in regression analysis and forecasting, and measures the standard deviation of the residuals arising between predicted and actual observed values for a modelling process. For our task of generating user movie ratings via recommendation algorithms.  

        """)
        st.latex(r"""RMSE = \sqrt{\frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui}\in \hat{R}}{(r_{ui}-\hat{r}_{ui})^2}}""")
        st.write("""
            
            Where R^ is the total number of recommendations generated for users and movies, with rui and r^ui being the true and predicted ratings for user u watching movie i respectively.

        """)
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    # ------------ EXPLORATORY DATA ANALYSIS ----------------------------
    if page_selection == "Exploratory Data Analysis":
        st.title("Data overview")
        st.write(data_overview)
        image_magic('https://phhp-faculty-cantrell.sites.medinfo.ufl.edu/files/2012/07/images-mod1-big-picture-eda.gif')
        st.write(description)
        data_visualisation(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        null_values_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        common_attributes_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        word_clouds_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        pub_year_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        movie_genre_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)
        budget_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score)



if __name__ == '__main__':
    main()
