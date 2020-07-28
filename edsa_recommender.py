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

import matplotlib.pyplot as plt
import seaborn as sns
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
        st.image('imgs/undraw_instant_information_6755.png', use_column_width=True)
        st.title("Solution Overview")
        st.markdown("""
            In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.
        """)
        st.write('## Problem Statment')
        st.write('To construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.')
        st.write('## Modelling')
        st.write('### 1. Collaborative Filtering (Singular Value Decomposition)')
        st.write('The collaborative filtering approach builds a model based on a user’s past behaviors (for example, items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in.')
        st.write('Within collaborative filtering, there are two well-known distinct approaches:')
        st.write('**Memory-Based:** models calculate the similarities between users / items based on user-item rating pairs.')
        st.write('**Model-Based:** models use some sort of machine learning algorithm to estimate the ratings.')
        st.write('### 2. Collaborative Filtering Using Cosine Similarity')
        st.write('This model uses cosine similarity to build the recommender system. Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space.')
        st.write('### 3. Content-Based Filtering (Linear Regression)')
        st.write('ontent-based recommenders treat recommendation as a user-specific classification problem and learn a classifier for the users likes and dislikes based on an items features. In this system, keywords are used to describe the items and a user profile is built to indicate the type of item this user likes.')
        st.write('Simple linear regression is a statistical method that shows the relationship between two continuous variables. This is represented by a straight line with the equation:')
        st.latex('$$ y = a + bx$$') 
        st.write('where $a$ is the intercept of the line with the y-axis, and $b$ is the gradient. The independent variable ($x$) is also known as the predictor and the dependent variable ($y$) is known as the target.')
        st.write('### 4. Collaborative Filtering (NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering)')
        st.latex(r'''
                 For this model, the data preparation is the same as it was for the SVD model, so we will skip to the modelling.

For this model, we tried different models and used the best performing one for this part of the section. 

* **Normal Predictor:**

Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal.The prediction $\hat{r}_{ui}$ is generated from a normal distribution $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$ where $\hat{\mu}$ and $\hat{\sigma}$ are estimated from the training data using Maximum Likelihood Estimation:

$$begin{split}\hat{\mu} &= \frac{1}{|R_{train}|} \sum_{r_{ui} \in R_{train}}
r_{ui}\\\\        \hat{\sigma} &= \sqrt{\sum_{r_{ui} \in R_{train}}
\frac{(r_{ui} - \hat{\mu})^2}{|R_{train}|}}\end{split}$$

* **KNNBaseline:**

A basic collaborative filtering algorithm taking into account a baseline rating.

The prediction $\hat{r}_{ui}$ is set as:

$$\hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{v \in N^k_i(u)}
\text{sim}(u, v) \cdot (r_{vi} - b_{vi})} {\sum\limits_{v \in
N^k_i(u)} \text{sim}(u, v)}$$

* **KNNBasic:**

A basic collaborative filtering algorithm.

The prediction $\hat{r}_{ui}$ is set as:

$$\hat{r}_{ui} = \frac{
\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot r_{vi}}
{\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)}$$


* **KNNWithMeans:**

A basic collaborative filtering algorithm, taking into account the mean ratings of each user.

The prediction $\hat{r}_{ui}$ is set as:

$$\hat{r}_{ui} = \mu_u + \frac{ \sum\limits_{v \in N^k_i(u)}
\text{sim}(u, v) \cdot (r_{vi} - \mu_v)} {\sum\limits_{v \in
N^k_i(u)} \text{sim}(u, v)}$$


* **KNNWithZScore:** 

A basic collaborative filtering algorithm, taking into account the z-score normalization of each user.

The prediction $\hat{r}_{ui}$ is set as:

$$ \hat{r}_{ui} = \mu_u + \sigma_u \frac{ \sum\limits_{v \in N^k_i(u)}
\text{sim}(u, v) \cdot (r_{vi} - \mu_v) / \sigma_v} {\sum\limits_{v
\in N^k_i(u)} \text{sim}(u, v)} $$ 

* **BaselineOnly:**

Algorithm predicting the baseline estimate for given user and item.

$$\hat{r}_{ui} = b_{ui} = \mu + b_u + b_i$$

If user $u$ is unknown, then the bias $b_u$ is assumed to be zero. The same applies for item $i$ with $b_i$.

* **CoClustering:**

A collaborative filtering algorithm based on co-clustering.

Basically, users and items are assigned some clusters Cu, Ci, and some co-clusters Cui.

The prediction $\hat{r}_{ui}$ is set as:

$$\hat{r}_{ui} = \overline{C_{ui}} + (\mu_u - \overline{C_u}) + (\mu_i
- \overline{C_i})$$


where $\overline{C_{ui}}$ is the average rating of co-cluster $C_{ui}$, $\overline{C_u}$ is the average rating of $u$’s cluster, and $\overline{C_u}$ is the average rating of $i$’s cluster. If the user is unknown, the prediction is $\hat{r}_{ui} = \mu_i$. If the item is unknown, the prediction is $\hat{r}_{ui} = \mu_u$. If both the user and the item are unknown, the prediction is $\hat{r}_{ui} = \mu$.
                 
                 
                 ''')
        
        # Kaggle scores
        rmse_svd = 0.82484
        lr_rmse = 1.06123
        baseline_rmse = 1.24856
        CF_cosine = 1.17123
        baselineOnly = 0.851014
        kNNBaseline = 0.860409
        kNNWithMeans = 0.861267
        kNNWithZScore = 0.861676
        kNNBasic = 0.909070
        coClustering = 0.919595
        normal = 1.299894

        # Compare RMSE squared values between models
        fig, caxis = plt.subplots(figsize=(12, 6))
        rmse_x = ['Linear Regression',
                  'BaselineOnly', 'SVD',
                  'Cosine Similarity',
                  'Baseline Only',
                  'KNN Baseline',
                  'KNN With Means',
                  'KNN WIth Z Score',
                  'KNN Basic',
                  'CoClustering',
                  'Normal Predictor']
        rmse_y = [lr_rmse, baseline_rmse, rmse_svd, CF_cosine, baselineOnly,
                  kNNBaseline, kNNWithMeans, kNNWithZScore, kNNBasic,
                  coClustering, normal]
        ax = sns.barplot(x=rmse_x, y=rmse_y, palette='plasma_r')
        plt.title('RMSE Values of Models', fontsize=14)
        plt.ylabel('RMSE')
        plt.xticks(rotation=90)
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(),
                    round(p.get_height(), 2),
                    fontsize=12, ha="center",
                    va='bottom')
        st.pyplot()
        st.write('## Conclusion')
        st.write('We were able to solve the problem statement and managed to build a model that estimates the ratings of the movie as well as make movies recommendations for the streamlit app. Based on the RMSE metric calculated for all models, we decided that the SVD is the best and would be our final model. It was definitely interesting to see that the collaborative model outperformed the content-based model because of the cold-start problem. The cold-start problem, which describes the difficulty of making recommendations when the users or the items are new, remains a great challenge for collaborative filtering. We ultimately chose the SVD model based on the RMSE metric as well as the runtime, as it only takes 15 minutes to run.')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    # ------------ EXPLORATORY DATA ANALYSIS ----------------------------
    if page_selection == "Exploratory Data Analysis":
        st.image('imgs/undraw_business_plan_5i9d.png', use_column_width=True)
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
