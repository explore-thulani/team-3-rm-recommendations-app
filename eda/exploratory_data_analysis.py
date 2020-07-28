from PIL import Image
import requests
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from wordcloud import WordCloud, STOPWORDS

import surprise
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split

import time

from surprise import SVD
from surprise import accuracy

import re

import streamlit as st

data_overview = """
    This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!

    For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.

    **Source:**
    The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB

    **Supplied Files:**
    * genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here
    * genome_tags.csv - user assigned tags for genome-related scores
    * imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.
    * links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.
    * sample_submission.csv - Sample of the submission format for the hackathon.
    * tags.csv - User assigned for the movies within the dataset.
    * test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.
    * train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.

    **Additional Information:**
    The below information is provided directly from the MovieLens dataset description files:

    * All ratings are contained in the file train.csv. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:
    userId,movieId,rating,timestamp
    The lines within this file are ordered first by userId, then, within user, by movieId.

    * Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

    * Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

    * All tags are contained in the file tags.csv. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:
    userId,movieId,tag,timestamp
    The lines within this file are ordered first by userId, then, within user, by movieId.
    Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.

    * Movie information is contained in the file movies.csv. Each line of this file after the header row represents one movie, and has the following format:
    movieId,title,genres
    Movie titles are entered manually or imported from https://www.themoviedb.org/, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

    * Genres are a pipe-separated list, and are selected from the following:
    Action
    Adventure
    Animation
    Children's
    Comedy
    Crime
    Documentary
    Drama
    Fantasy
    Film-Noir
    Horror
    Musical
    Mystery
    Romance
    Sci-Fi
    Thriller
    War
    Western
    (no genres listed)
    Links Data File Structure (links.csv)
    Identifiers that can be used to link to other sources of movie data are contained in the file links.csv. Each line of this file after the header row represents one movie, and has the following format:

    * movieId is an identifier for movies used by https://movielens.org. E.g., the movie Toy Story has the link https://movielens.org/movies/1.

    * imdbId is an identifier for movies used by http://www.imdb.com. E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.

    * tmdbId is an identifier for movies used by https://www.themoviedb.org. E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.

    * As described in this article, the tag genome encodes how strongly movies exhibit particular properties represented by tags (atmospheric, thought-provoking, realistic, etc.). The tag genome was computed using a machine learning algorithm on user-contributed content including tags, ratings, and textual reviews.

    * The genome is split into two files. The file genome-scores.csv contains movie-tag relevance data in the following format:

    * The second file, genome-tags.csv, provides the tag descriptions for the tag IDs in the genome file, in the following format:
    tagId,tag

    ## Exploratory Data Analysis And Insights
"""

description = """
Exploratory Data Analysis (EDA) is a fundamental part of the Machine Learning process. The data is analysed in order to extract information that a model may overlook. In this section, we will summarise the main characteritics of the data and also look into the sentiment classes provided in our training datasets.
"""

def image_magic(image_url):
    open_door = requests.get(image_url)
    climate_picture = Image.open(BytesIO(open_door.content))
    st.image(climate_picture, use_column_width = True)

def data_visualisation(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):

    train_user = pd.DataFrame(train_df['userId'].value_counts()).reset_index()
    train_user.rename(columns = {'index':'userId','userId' : 'count'}, inplace = True)
    st.table(train_user.head())

    # --------------------- COMMON USERS ----------------------------------
    st.write("### 5.1. Visualising Common Users")
    ax = sns.barplot(train_user[train_user['count']>1000]['userId'].values,
                 train_user[train_user['count']>1000]['count'].values)
    ax.set(xlabel ='userId', ylabel ='count')
    sns.set(rc={'figure.figsize':(10,10)})
    plt.title('Common Users')
    st.pyplot()

    # --------------------- COMMON DISTRIBUTION ---------------------------
    st.write("""### 5.2. Visualising Rating Distribution""")
    train_rating = pd.DataFrame(train_df['rating'].value_counts()).reset_index()
    train_rating.rename(columns = {'index':'rating','rating' : 'count'}, inplace = True)
    train_rating
    ax = sns.barplot(train_rating['rating'].values,
                     train_rating['count'].values)
    ax.set(xlabel ='rating', ylabel ='count')
    sns.set(rc={'figure.figsize':(10,10)})
    plt.title('Rating Distribution')
    st.pyplot()

    # --------------------- VISUALISING THE DATAFRAME ---------------------
    st.write("### 5.3. Visualising The Dataframe")
    dataframes = ['train_df','test_df','tags_df','imdb_df','links_df','movies_df','genome_tags','genome_score']
    sizes = [len(train_df), len(test_df), len(tags_df), len(imdb_df), len(links_df), len(movies_df),
            len(genome_tags), len(genome_score)]
    total_size_df = pd.DataFrame(list(zip(dataframes, sizes)),
                   columns =['dataframe', 'sizes'])
    st.table(total_size_df)
    explodeTuple = (0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0)
    fig1, ax1 = plt.subplots()
    ax1.pie(total_size_df['sizes'].values,
            labels=total_size_df['dataframe'].values,
            explode=explodeTuple,
            startangle=90,autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot()

    len_list = [['train_df', len(train_df)],['tags_df', len(tags_df)],['imdb_df', len(imdb_df)],
            ['links_df',len(links_df)],['movies_df', len(movies_df)],['genome_tags', len(genome_tags)],
            ['genome_score', len(genome_score)]]
    len_df = pd.DataFrame(len_list,columns = ['Dataset','Size'])
    ax = sns.barplot(len_df['Dataset'].values, len_df['Size'].values)
    ax.set(xlabel ='Dataset', ylabel ='Dataset_size(in 10 millions)')
    sns.set(rc={'figure.figsize':(10,10)})
    plt.title('Dataset_comparison')
    st.pyplot()

def null_values_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    train_count = pd.DataFrame(train_df.isnull().sum())
    test_count = pd.DataFrame(test_df.isnull().sum())
    tags_count = pd.DataFrame(tags_df.isnull().sum())
    movies_count = pd.DataFrame(movies_df.isnull().sum())
    links_count = pd.DataFrame(links_df.isnull().sum())
    imdb_count = pd.DataFrame(imdb_df.isnull().sum())
    genomet_count = pd.DataFrame(genome_tags.isnull().sum())
    genomes_count = pd.DataFrame(genome_score.isnull().sum())

    # ---------------- VISUALISING NULL VALUES FOR EACH DATAFRAME -------------
    st.write("### 5.4. Visualising Null Values For Each Dataframe")
    st.write("""
        Missing values, mainly known as null values, occur due to multiple reasons including errors whilst collecting data. These values are removed because they add no value to the output of the models; all data has to be valid and valued. Duplicates are also removed as they do not provide any new information. The reduced number of values also result in less model run time.
    """)

    plt.bar(tags_count.index,tags_count.values.reshape(len(tags_count),),color='red')
    plt.xlabel('column_name')
    plt.ylabel('count')
    plt.title('Tags Dataframe')
    st.pyplot()
    #tags_df[tags_df.isnull().any(axis=1)]

    plt.bar(links_count.index,links_count.values.reshape(len(links_count),),color='green')
    plt.xlabel('column_name')
    plt.ylabel('count')
    plt.title('Links Dataframe')
    st.pyplot()

    plt.bar(imdb_count.index,imdb_count.values.reshape(len(imdb_count),),color='orange')
    plt.xlabel('column_name')
    plt.ylabel('count')
    plt.title('Imdb Dataframe')
    st.pyplot()


def common_attributes_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    st.write("""
    ### 5.5. Visualising Common Attributes/Correlation For Each Dataframe

    In statistics, correlation or dependence is any statistical relationship, whether causal or not, between two random variables or bivariate data. In the broadest sense correlation is any statistical association, though it commonly refers to the degree to which a pair of variables are linearly related. The correlation coefficient is measured on a scale that varies from + 1 through 0 to - 1. Complete correlation between two variables is expressed by either + 1 or -1. When one variable increases as the other increases the correlation is positive; when one decreases as the other increases it is negative. Complete absence of correlation is represented by 0.
    """)

    corr1 = pd.concat([train_df, tags_df], axis=1).corr()
    st.table(corr1.head())

    ax = sns.heatmap(corr1, vmin=0, vmax=1)
    common_at = pd.DataFrame(train_df['userId'].isin(tags_df['userId']).value_counts())
    plt.bar(['False','True'],common_at.values.reshape(len(common_at),),color='red')
    plt.xlabel('Column Name')
    plt.ylabel('Count in Millions')
    plt.title('Common Attributes')
    st.pyplot()

    st.write("""
        As seen above, not many rows have common attributes we can use to link the train_df dataframe with the tags_df dataframe. Knowing the user and the type of genre they prefer to view, we can make a calculated estimate based on their watch history. Finding a link between the tables can increase the amount of variables used to predict the rating based on the user preferences.
    """)
    common_at2 = pd.DataFrame(train_df['movieId'].isin(tags_df['movieId']).value_counts())
    plt.bar(['False','True'],common_at2.values.reshape(len(common_at2),),color='blue')
    plt.xlabel('Column Name')
    plt.ylabel('Count in Millions')
    plt.title('Common Attributes')
    st.pyplot()

    common_at = pd.DataFrame(train_df['timestamp'].isin(tags_df['timestamp']).value_counts())
    plt.bar(['False','True'],common_at.values.reshape(len(common_at),),color='orange')
    plt.xlabel('Column Name')
    plt.ylabel('Count in Millions')
    plt.title('Common Attributes')
    st.pyplot()

    common_at1 = pd.DataFrame(movies_df['movieId'].isin(links_df['movieId']).value_counts())
    plt.bar(['True'],common_at1.values.reshape(len(common_at1),),color='green')
    plt.xlabel('Column Name')
    plt.ylabel('Count')
    plt.title('Common Attributes')
    st.pyplot()


def word_clouds_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    st.write("""
        ### 5.6. Word Clouds

        Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites.
    """)
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in tags_df['tag']:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    st.pyplot()

    value_count = pd.DataFrame(tags_df['tag'].value_counts()).reset_index()
    value_count.rename(columns = {'index':'genre','tag' : 'count'}, inplace = True)
    st.table(value_count.head())
    st.table(value_count[value_count['count']>2500]['count'].values)

    explodeTuple = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2)
    fig1, ax1 = plt.subplots()
    ax1.pie(value_count[value_count['count']>2500]['count'].values,
            labels=value_count[value_count['count']>2500]['genre'].values,autopct='%1.1f%%',startangle=90)
    ax1.axis('equal')
    plt.figure(figsize = (8, 8))
    st.pyplot()

    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in genome_tags['tag']:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    st.pyplot()


def pub_year_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    st.write("""### 5.7. Publish Years Exploration""")
    dates=[]
    for title in movies_df['title']:
        if title[-1] == " ":
            year = title[-6:-2]
            try:dates.append(int(year))
            except: dates.append(9999)

        else:
            year = title[-5:-1]
            try:dates.append(int(year))
            except: dates.append(9999)

    movies_df['Publish Year'] = dates
    movies_df['Publish Year'].unique()

    fig, ax = plt.subplots(figsize=(14, 7))
    dataset = movies_df[movies_df['Publish Year']!= 9999]
    new = dataset[dataset['Publish Year']!=6]
    plt.hist(new['Publish Year'], bins=10, color='#3498db', ec='#2980b9')
    plt.xlabel('Publish Year', size=20)
    plt.ylabel('Count', size=20)
    plt.title('Movies Released Per Year', size=25)
    st.pyplot()


def movie_genre_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    st.write("""### 5.8. Movie Genre Exploration""")
    genres = pd.DataFrame(movies_df['genres'].str.split("|").tolist(), index=movies_df['movieId']).stack()
    genres = genres.reset_index([0, 'movieId'])
    genres.columns = ['movieId', 'Genre']
    st.table(genres.head())

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.countplot(x='Genre',data=genres,palette='CMRmap', order=genres['Genre'].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('Genre', size=20)
    plt.ylabel('Count', size=20)
    plt.title('Distribution of Movie Genres', size=25)
    st.pyplot()


def budget_viz(train_df,test_df,tags_df,imdb_df,links_df,movies_df,genome_tags,genome_score):
    st.write("""### 5.9. Budget Visualisation""")
    new_l = list(imdb_df['budget'])
    imdb_df['runtime'] = imdb_df['runtime'].fillna(imdb_df['runtime'].mean())
    imdb_df.isnull().sum()
    st.table(imdb_df.head(1))

    imdb_df['budget'] = imdb_df['budget'].str.replace('[\,]', '', regex=True)
    def clean_txt(text):
        text = re.sub(r'[0-9]+',"",str(text))
        return text

    imdb_df['currency'] = imdb_df['budget'].apply(clean_txt)
    st.table(imdb_df.head(1))

    currencies =  list(imdb_df['currency'])
    new_series = pd.Series(currencies).value_counts()
    currencies_count_df = pd.DataFrame(new_series).reset_index()
    st.table(currencies_count_df.head())
    percentage = currencies_count_df[0]/len(imdb_df)

    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(aspect="equal"))

    data = currencies_count_df[0]
    currencies = currencies_count_df['index']

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    ax.legend(wedges, currencies,
              title="currencies",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=15, weight="bold")

    ax.set_title("Currency Counts")

    st.pyplot()
