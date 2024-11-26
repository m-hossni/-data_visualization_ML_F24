import os
os.environ["OMP_NUM_THREADS"] = "1"

import io
import streamlit as st

from st_aggrid import AgGrid, GridOptionsBuilder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import plotly
import altair as alt



def load_data(filename):
    return pd.read_csv(filename)

# Now load the data using the function
data = load_data('netflix_titles.csv')

def view_our_data(data, n=5):
    #print(data.head(n))  # Display only the first n rows
    st.write(data.head(n))

def view_our_data_info(data):
    st.write("### Dataset Info for Netflix Dataset")
    
    # Capture data.info() output
    with io.StringIO() as buffer:
        data.info(buf=buffer)  # Write info to the buffer
        info_str = buffer.getvalue()  # Retrieve the string from the buffer
    
    # Display the captured info in a code block
    st.code(info_str, language="plaintext")
def view_our_data_description(data):
    st.write("### Netflix Dataset Description")
    
    # Get the descriptive statistics of the data
    desc = data.describe(include='all').transpose()  # Transpose for a cleaner view
    
    # Configure AgGrid options
    grid_options = GridOptionsBuilder.from_dataframe(desc)
    grid_options.configure_default_column(editable=False, groupable=True)
    grid_options.configure_pagination(paginationAutoPageSize=True)
    
    # Render the AgGrid interactive table
    AgGrid(
        desc.reset_index().rename(columns={"index": "Column Name"}),  # Add "Column Name" as a separate column
        gridOptions=grid_options.build(),
        height=400,
        theme="streamlit",
    )
    
def view_our_data_columns(data):
    st.write("### Netflix Data Columns and Information")
    
    # Create a DataFrame with column details
    col_info = pd.DataFrame({
        "Column Name": data.columns,
        "Null Count": [data[col].isna().sum() for col in data.columns],
        "Total Records": [len(data[col]) for col in data.columns]
    })
    
    # Configure AgGrid options
    grid_options = GridOptionsBuilder.from_dataframe(col_info)
    grid_options.configure_default_column(editable=False, groupable=True)
    grid_options.configure_column("Column Name", headerCheckboxSelection=True)
    grid_options.configure_selection("single")
    grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    
    # Render AgGrid interactive table
    AgGrid(
        col_info,
        gridOptions=grid_options.build(),
        height=400,
        theme="streamlit",  # Corrected theme
    )
    
# # Chart 1: Distribution of TV Shows and Movies by Type - Paige
# 
# *   Use ML algorithm to analyse the distribution of TV shows and movies.
# 
# * Visualize the proportion using a pie chart, where each slice represents the percentage of TV shows and movies.

def view_tv_vs_movie(data):
    # Calculate the distribution of TV shows vs Movies
    tv_mov_dist = data['type'].value_counts()

    # Plot interactive pie chart
    fig = px.pie(tv_mov_dist,
                 values=tv_mov_dist.values,
                 names=tv_mov_dist.index,
                 title="Distribution of TV Shows vs Movies on Netflix",
                 hole=0.5,
                 color_discrete_sequence=['Skyblue', 'Blue'])  # Cohesive color map for all plots

    # Update title font size
    fig.update_layout(
        title_font_size=24  # Increase the title font size
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
# # Chart 2: Relationship between Release Year and Duration  - Mohamed
# 
# * Perform predictive analysis to understand the relationship between release year and duration of TV shows or movies using machine learning regression techniques. **[note-M: the data does not seem to have trend :/]**
# 
# * Develop a predictive model to estimate the duration based on the release year.
# 
# * Create a scatter plot visualization to depict the relationship between release year and duration.

# # Dropping empty values for analysis

data_dropna = data.dropna().reset_index()
data_dropna


# # Adding new column for unique genres
# 
# spliting the liste_in column and removeing spaces to find unique genres. this was done to see if there is any trend between duration and year of release based on genre

#spliting genres
genres_seprated_movies = []
genres_seprated_shows = []
genres_seprated = []
for i, genres in enumerate(data_dropna['listed_in']):
  generes_split = genres.split(',')
  for genre in generes_split:
      if data_dropna['type'][i] == 'Movie' and genre.strip() not in genres_seprated_movies:
        genres_seprated_movies.append(genre.strip())
      if data_dropna['type'][i] == 'TV Show' and genre.strip() not in genres_seprated_shows:
        genres_seprated_shows.append(genre.strip())
      if genre.strip() not in genres_seprated:
        genres_seprated.append(genre.strip())

print(len(genres_seprated_movies))
print(len(genres_seprated_shows))
print(genres_seprated)
data_genre = data_dropna.copy()
data_genre['genre'] = data_genre['listed_in'].apply(lambda x : [item.strip() for item in x.split(',')])
data_genre = data_genre.explode('genre')
data_genre


# # creating a numerical value for duration in a new column

def get_duration_val(duration):
  return int(duration.split()[0])
data_genre['new_duration'] = data_genre['duration'].apply(get_duration_val)
data_genre


# # Analyzing duration difference between movie genres and showing the number of movies for each


movies_list = data_genre[data_genre['type'] == 'Movie']
duration_genre = movies_list.groupby('genre').agg(
    average_duration =('new_duration', 'mean'),
    count = ('new_duration', 'count')
).reset_index().sort_values(by='average_duration')

plt.figure(figsize=(6,4))
plot = sns.barplot(duration_genre,x='genre',y='average_duration')
plt.xticks(rotation=45, ha='right')
for i, row in duration_genre.iterrows():
  plot.text(row.genre,row['average_duration']+2, str(row['count']),color='black', ha='center', rotation=90)
plt.ylabel('Average Movie Duration (min)')
plt.xlabel('Movie Genre')
plt.title('The movie genres and their average duration labeled with the number of movies in that genre')
plt.show()


# # overview for movie duration vs release year for each movie genre


unique_genres = movies_list['genre'].unique()
num_genre = len(unique_genres)
fig, axes = plt.subplots(nrows=(num_genre+1)//2, ncols=2, figsize=(15, ((num_genre+1)//2) * 5))
axes = axes.flatten()
for i in range(num_genre):
  sns.scatterplot(data=movies_list[movies_list['genre']== unique_genres[i]],x='release_year',y='new_duration', ax=axes[i])
  axes[i].set_title('Duration of Movies by release year [genre = {}]'.format(movies_list['genre'].unique()[i]))
  axes[i].set_xlabel('release year')
  axes[i].set_ylabel('duration (min)')
plt.show()


# # from the above, we can use regression based on three columns [genre, release_year, new_duration] with new_duration as the target for the predictive analysis given genre and release_year
# 

#converting categorical data into numbers
movies_transformed = movies_list.copy()
genre_dict = {}
for i, genre in enumerate(unique_genres):
  genre_dict[genre] = i

print(genre_dict)
movies_transformed['genre_code'] = movies_transformed['genre'].map(genre_dict)
#finding mean and std for every column to normalize
mean_release_year = movies_transformed['release_year'].mean()
std_release_year = movies_transformed['release_year'].std()
mean_duration = movies_transformed['new_duration'].mean()
std_duration = movies_transformed['new_duration'].std()
mean_genre = movies_transformed['genre_code'].mean()
std_genre = movies_transformed['genre_code'].std()
movies_transformed_normalized = movies_transformed.copy().reset_index()
movies_transformed_normalized['release_year'] = (movies_transformed_normalized['release_year'] -  mean_release_year) / std_release_year
movies_transformed_normalized['new_duration'] = (movies_transformed_normalized['new_duration'] -  mean_duration) / std_duration
movies_transformed_normalized['genre_code'] = (movies_transformed_normalized['genre_code'] -  mean_genre) / std_genre

#targets and inputs splitting
X = movies_transformed_normalized[['release_year'] +['genre_code']]
y = movies_transformed_normalized['new_duration']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(mse)


# # visualizing predicted results

predictions = (model.predict(X)*std_duration) +mean_duration
movies_list['predicted_duration'] = predictions
fig, axes = plt.subplots(nrows=(num_genre+1)//2, ncols=2, figsize=(15, ((num_genre+1)//2) * 5))
axes = axes.flatten()
for i in range(num_genre):
  sns.scatterplot(data=movies_list[movies_list['genre']== unique_genres[i]],x='release_year',y='new_duration', ax=axes[i])
  sns.lineplot(data=movies_list[movies_list['genre']== unique_genres[i]],x='release_year',y='predicted_duration', ax=axes[i], color='blue', label='Predicted')
  axes[i].set_title('Duration of Movies by release year [genre = {}]'.format(movies_list['genre'].unique()[i]))
  axes[i].set_xlabel('release year')
  axes[i].set_ylabel('duration (min)')
plt.show()


# # doing the same with rating (maybe we can add it as a third input)
duration_genre = movies_list.groupby('rating').agg(
    average_duration =('new_duration', 'mean'),
    count = ('new_duration', 'count')
).reset_index().sort_values(by='average_duration')

plt.figure(figsize=(10,6))
plot = sns.barplot(duration_genre,x='rating',y='average_duration')
plt.xticks(rotation=45, ha='right')
for i, row in duration_genre.iterrows():
  plot.text(row.rating,row['average_duration']+2, str(row['count']),color='black', ha='center', rotation=90)
plt.ylabel('Average Movie Duration (min)')
plt.xlabel('Movie rating')
plt.title('The movie rating and their average duration labeled with the number of movies in that genre')
plt.show()


unique_ratings = movies_list['rating'].unique()
num_ratings = len(unique_ratings)
fig, axes = plt.subplots(nrows=(num_ratings+1)//2, ncols=2, figsize=(15, ((num_ratings+1)//2) * 5))
axes = axes.flatten()
for i in range(num_ratings):
  sns.scatterplot(data=movies_list[movies_list['rating']== unique_ratings[i]],x='release_year',y='new_duration', ax=axes[i])
  axes[i].set_title('Duration of Movies by release year [rating = {}]'.format(movies_list['rating'].unique()[i]))
  axes[i].set_xlabel('release year')
  axes[i].set_ylabel('duration (min)')
plt.show()


#ploting duration vs release year for refrence
def get_duration_val(duration):
  return int(duration.split()[0])
data['new_duration'] = data.dropna()['duration'].apply(get_duration_val)
sns.scatterplot(data=data[data['type'] == 'Movie'],x='release_year',y='new_duration')
plt.legend(loc='upper right', bbox_to_anchor=(1.25,1),title='Movie rating')
plt.title('Duration of Movies by release year')
plt.xlabel('release year')
plt.ylabel('duration (min)')
plt.show()


sns.scatterplot(data=data[data['type'] == 'TV Show'],x='release_year',y='new_duration')
plt.legend(loc='upper right', bbox_to_anchor=(1.25,1),title='Show rating')
plt.title('Duration of TV Shows by release year')
plt.xlabel('release year')
plt.ylabel('duration (Seasons)')
plt.show()

####UPDATED MOHAMED CODE FOR HIS CHARTS ######################
#############################################################
def page_Chart_1():
    tv_mov_dist = pd.DataFrame(data['type'].value_counts())
    tv_mov_dist = tv_mov_dist.reset_index()

    base = alt.Chart(tv_mov_dist).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='count', type="quantitative", stack=True),
        color=alt.Color(field='type', type="nominal"),
    ).transform_joinaggregate(
        TotalCount='sum(count)',
    ).transform_calculate(
        PercentOfTotal="datum.count / datum.TotalCount"
    )
    pie = base.mark_arc(innerRadius=80)
    text = base.mark_text(radius=200, fill="black", size=20).encode(
        alt.Text(field='PercentOfTotal', type="quantitative", format=".0%")
    ) 
    donut_chart = pie + text
    return donut_chart

def page_Chart_2():
    data_dropna = data.dropna().reset_index()
    # Additional preprocessing steps

    # Model for predicting duration
    movies_list = data_dropna[data_dropna['type'] == 'Movie']
    duration_genre = movies_list.groupby('genre').agg(
        average_duration=('new_duration', 'mean'),
        count=('new_duration', 'count')
    ).reset_index().sort_values(by='average_duration')
    
    model = LinearRegression()
    X = movies_list[['release_year', 'genre_code']]
    y = movies_list['new_duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    predictions = model.predict(X)
    movies_list['predicted_duration'] = predictions
    return mse, duration_genre, movies_list

def page_Chart_3():
    data_shows = data[data['type'] == 'TV Show']
    grp_data = data_shows.groupby('country').agg({
        'title': 'count'
    }).reset_index()
    grp_data.columns = ['Country', 'Total Shows']
    filtered_data = grp_data[grp_data['Total Shows'] > 5].sort_values(by='Total Shows')
    
    bar_plot_movies = alt.Chart(filtered_data).encode(x=alt.X('Country').sort('-y')).mark_bar().encode(y='Total Shows').interactive()
    return bar_plot_movies

def page_Chart_4():
    tv_shows = data[(data['type'] == 'TV Show') & (~data['rating'].fillna('').str.contains('min'))]
    movies = data[(data['type'] == 'Movie') & (~data['rating'].fillna('').str.contains('min'))]
    tv_rating_counts = tv_shows['rating'].value_counts()
    movie_rating_counts = movies['rating'].value_counts()
    tv_rating_counts_df = pd.DataFrame(tv_rating_counts).reset_index()
    tv_rating_counts_df.columns = ['Rating', 'Count']
    movie_rating_counts_df = pd.DataFrame(movie_rating_counts).reset_index()
    movie_rating_counts_df.columns = ['Rating', 'Count']
    
    return tv_rating_counts_df, movie_rating_counts_df

#############################################################
#############################################################


# # Chart 3: Distribution of Content by Country  - Mohamed
# 
# * Aggregate TV shows and movies by country using ML technique. **no need for ML**
# 
# * Visualize the distribution using a bar chart, where each bar represents a country and the height represents the number of TV shows or movies produced.
def view_tv_country(data):
    data_shows = data[data['type'] == 'TV Show']
    grp_data = data_shows.groupby('country').agg({
        'title': 'count'
    }).reset_index()
    grp_data.columns = ['Country', 'Total Shows']
    filtered_data = grp_data[grp_data['Total Shows'] > 5].sort_values(by='Total Shows')

    # Create the plot
    plt.figure(figsize=(16, 12))
    plt.bar(filtered_data['Country'], filtered_data['Total Shows'])
    plt.xlabel('Countries with More Than 5 TV Shows', fontsize=14)  # Adjust font size
    plt.ylabel('Number of Shows', fontsize=14)  # Adjust font size
    plt.title('Netflix Dataset: Distribution of Countries With More Than 5 TV Shows', fontsize=18)  # Larger title font size
    plt.xticks(rotation=60, ha='right', fontsize=12)  # Adjust rotation angle to 60 degrees for better readability

    # Display the plot using Streamlit
    st.pyplot(plt)

def ratings_movie_country():
    # Ensure view_tv_movie_rating_distribution() correctly returns data
    tv_shows, movies = view_tv_movie_rating_distribution(data)
    
    # Call the function that plots the distribution of movies by country
    view_movie_country(movies)  # Pass 'movies' data directly to view_movie_country

def view_movie_country(data_Movies):
    # Group movies by country
    grp_data_movies = data_Movies.groupby('country').agg({
        'title': 'count'
    }).reset_index()
    grp_data_movies.columns = ['Country', 'Total Movies']
    
    # Filter countries with more than 10 movies
    filtered_data = grp_data_movies[grp_data_movies['Total Movies'] > 10].sort_values(by='Total Movies')

    # Create the plot
    plt.figure(figsize=(18, 12))
    plt.bar(filtered_data['Country'], filtered_data['Total Movies'], color='skyblue')
    plt.xlabel('Countries with More Than 10 Movies', fontsize=18)
    plt.ylabel('Number of Movies', fontsize=18)
    plt.title('Netflix Dataset: Countries With More Than 10 Movies', fontsize=24)
    
    # Rotate x-axis labels by 90 degrees
    plt.xticks(rotation=90, ha='center', fontsize=16)

    # Display the plot using Streamlit
    st.pyplot(plt)

# # Chart 4: Distribution of Content by Rating - Jack
# 
# * Analyse the distribution of TV shows and movies by rating using ML clustering techniques.
# 
# * Visualize the distribution using a bar chart, where each bar represents a rating category (e.g., G, PG, PG-13, R, etc.) and the height represents the number of TV shows or movies in each category.

# Filter dataset to exclude ratings with time measurements
# data['type'] == 'TV Show': This checks each row in the dataset to see if the value in the "type" column is "TV Show".
# data['rating'].fillna('').str.contains('min'): This checks the "rating" column to see if it contains the substring "min".
# Before doing this check, it replaces any missing values (NaN) in the "rating" column with an empty string ('') to avoid errors.
# ~ (tilde): This operator negates the boolean values, meaning it converts True to False and vice versa.
# So, it is filtering out any ratings that do contain "min". & (ampersand): This operator combines both conditions,
# so we only keep rows where the type is "TV Show" and the rating does not contain "min".
# tv_shows =: This creates a new variable called tv_shows that stores the filtered results.

def view_tv_movie_rating_distribution(data):
    # Filter TV Shows and Movies
    tv_shows = data[data['type'] == 'TV Show']
    movies = data[data['type'] == 'Movie']
    
    # Get rating counts for TV Shows and Movies
    tv_rating_counts = tv_shows['rating'].value_counts()
    movie_rating_counts = movies['rating'].value_counts()
    
    # Create DataFrames for ratings counts
    tv_rating_counts_df = pd.DataFrame(tv_rating_counts).reset_index()
    movie_rating_counts_df = pd.DataFrame(movie_rating_counts).reset_index()
    tv_rating_counts_df.columns = ['Rating', 'Count']
    movie_rating_counts_df.columns = ['Rating', 'Count']
    
    # Combine TV Shows and Movie ratings counts into one DataFrame for comparison
    ratings_combined = pd.DataFrame({'TV Shows': tv_rating_counts, 'Movies': movie_rating_counts}).fillna(0)

    # Plot the rating distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # TV Show ratings plot
    axes[0].bar(tv_rating_counts_df['Rating'], tv_rating_counts_df['Count'], color='skyblue')
    axes[0].set_title('Distribution of Ratings for TV Shows')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Number of TV Shows')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Movie ratings plot
    axes[1].bar(movie_rating_counts_df['Rating'], movie_rating_counts_df['Count'], color='orange')
    axes[1].set_title('Distribution of Ratings for Movies')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Number of Movies')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Display the plots in Streamlit
    st.pyplot(fig)
    
    # Return combined ratings for potential use elsewhere
    return ratings_combined


def view_movie_ratings_by_country(data):
    # Filter movies from the dataset
    movies = data[data['type'] == 'Movie']
    
    # Filter ratings between "PG" and "TV-Y17"
    valid_ratings = ["PG", "TV-Y7", "TV-Y7-FV", "PG-13", "TV-PG", "TV-14", "TV-MA", "TV-Y", "TV-G", "TV-Y7", "TV-Y17"]
    movies = movies[movies['rating'].isin(valid_ratings)]
    
    # Group data by country and rating, and count the number of movies for each combination
    grouped = movies.groupby(['country', 'rating']).size().reset_index(name='Count')
    
    # Filter out countries with no ratings
    grouped = grouped[grouped['country'].notnull()]
    
    # Select top N countries by the total number of movies for better visualization
    top_countries = grouped.groupby('country')['Count'].sum().nlargest(10).index
    grouped = grouped[grouped['country'].isin(top_countries)]
    
    # Create the interactive bar chart
    fig = px.bar(
        grouped,
        x="rating",
        y="Count",
        color="country",
        title="Movie Ratings Distribution by Country (Top 10 Countries)",
        labels={"rating": "Movie Rating", "Count": "Number of Movies", "country": "Country"},
        barmode="group",
        height=600,
        width=2000,  # Extra-wide chart for more space
    )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(title="Rating", tickangle=45),
        yaxis=dict(title="Number of Movies"),
        legend=dict(
            title="Country",
            orientation="h",  # Horizontal legend
            yanchor="top",
            xanchor="right",
            x=1.0,  # Positioned at the top-right
            y=1.0,
            font=dict(size=10),  # Smaller legend font
        ),
        font=dict(size=14),
        bargap=0.01,  # Very minimal gap to make bars as wide as possible
    )
    
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)


def view_tv_show_ratings_by_country(data):
    # Filter TV shows from the dataset
    tv_shows = data[data['type'] == 'TV Show']
    
    # Filter ratings between "PG" and "TV-Y17"
    valid_ratings = ["PG", "TV-Y7", "TV-Y7-FV", "PG-13", "TV-PG", "TV-14", "TV-MA", "TV-Y", "TV-G", "TV-Y7", "TV-Y17"]
    tv_shows = tv_shows[tv_shows['rating'].isin(valid_ratings)]
    
    # Group data by country and rating, and count the number of TV shows for each combination
    grouped = tv_shows.groupby(['country', 'rating']).size().reset_index(name='Count')
    
    # Filter out countries with no ratings
    grouped = grouped[grouped['country'].notnull()]
    
    # Select top N countries by the total number of TV shows for better visualization
    top_countries = grouped.groupby('country')['Count'].sum().nlargest(10).index
    grouped = grouped[grouped['country'].isin(top_countries)]
    
    # Create the interactive bar chart
    fig = px.bar(
        grouped,
        x="rating",
        y="Count",
        color="country",
        title="TV Show Ratings Distribution by Country (Top 10 Countries)",
        labels={"rating": "TV Show Rating", "Count": "Number of TV Shows", "country": "Country"},
        barmode="group",
        height=600,
        width=2000,  # Extra-wide chart for more space
    )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(title="Rating", tickangle=45),
        yaxis=dict(title="Number of TV Shows"),
        legend=dict(
            title="Country",
            orientation="h",  # Horizontal legend
            yanchor="top",
            xanchor="right",
            x=1.0,  # Positioned at the top-right
            y=1.0,
            font=dict(size=10),  # Smaller legend font
        ),
        font=dict(size=14),
        bargap=0.01,  # Very minimal gap to make bars as wide as possible
    )
    
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)
    
    
# def plot_movie_ratings(movies):
#     # Create an interactive bar chart for the distribution of movies by rating
#     fig = px.histogram(movies, 
#                        x="rating", 
#                        title="Distribution of Movies by Rating",
#                        labels={"rating": "Movie Rating"},
#                        color="rating",  # Color by rating to differentiate them
#                        category_orders={"rating": sorted(movies['rating'].unique())},  # Sort ratings
#                        marginal="box",  # Add box plot for better understanding of the distribution
#                        histnorm="percent")  # Normalize the histogram to show percentages
    
#     fig.update_layout(
#         xaxis_title="Movie Rating",
#         yaxis_title="Percentage",
#         template="plotly_dark",  # Use a dark theme or choose a different one
#         showlegend=False  # Hide legend to avoid clutter
#     )
    
#     return fig

# def plot_tv_ratings(tv_shows):
#     # Create an interactive bar chart for the distribution of TV Shows by rating
#     fig = px.histogram(tv_shows, 
#                        x="rating", 
#                        title="Distribution of TV Shows by Rating",
#                        labels={"rating": "TV Show Rating"},
#                        color="rating",  # Color by rating to differentiate them
#                        category_orders={"rating": sorted(tv_shows['rating'].unique())},  # Sort ratings
#                        marginal="box",  # Add box plot for better understanding of the distribution
#                        histnorm="percent")  # Normalize the histogram to show percentages
    
#     fig.update_layout(
#         xaxis_title="TV Show Rating",
#         yaxis_title="Percentage",
#         template="plotly_dark",  # Use a dark theme or choose a different one
#         showlegend=False  # Hide legend to avoid clutter
#     )
    
#     return fig

# def plot_movie_distribution(movies):
#     # Create an interactive bar chart for the distribution of movies by rating
#     fig = px.histogram(movies, 
#                        x="rating", 
#                        title="Distribution of Movies by Rating",
#                        labels={"rating": "Movie Rating"},
#                        color="rating",  # Color by rating to differentiate them
#                        category_orders={"rating": sorted(movies['rating'].unique())},  # Sort ratings
#                        marginal="box",  # Add box plot for better understanding of the distribution
#                        histnorm="percent")  # Normalize the histogram to show percentages
    
#     fig.update_layout(
#         xaxis_title="Movie Rating",
#         yaxis_title="Percentage",
#         template="plotly_dark",  # Use a dark theme or choose a different one
#         showlegend=False  # Hide legend to avoid clutter
#     )
#     st.plotly_chart(fig)

# def plot_tv_distribution(tv_shows):
#     # Create an interactive bar chart for the distribution of TV Shows by rating
#     fig = px.histogram(tv_shows, 
#                        x="rating", 
#                        title="Distribution of TV Shows by Rating",
#                        labels={"rating": "TV Show Rating"},
#                        color="rating",  # Color by rating to differentiate them
#                        category_orders={"rating": sorted(tv_shows['rating'].unique())},  # Sort ratings
#                        marginal="box",  # Add box plot for better understanding of the distribution
#                        histnorm="percent")  # Normalize the histogram to show percentages
    
#     fig.update_layout(
#         xaxis_title="TV Show Rating",
#         yaxis_title="Percentage",
#         template="plotly_dark",  # Use a dark theme or choose a different one
#         showlegend=False  # Hide legend to avoid clutter
#     )
#     st.plotly_chart(fig)

def view_tv_movie_rating_distribution(data):
    # Filter and separate tv_shows and movies from the dataset
    tv_shows = data[data['type'] == 'TV Show']
    movies = data[data['type'] == 'Movie']
    
    # Return both tv_shows and movies
    return tv_shows, movies





def view_tv_movie_rating_distribution(data):
    # Filter the dataset into TV Shows and Movies
    tv_shows = data[data['type'] == 'TV Show']
    movies = data[data['type'] == 'Movie']
    
    # If you want the ratings grouped by the 'rating' column, you can aggregate here, but it should still return DataFrames
    tv_shows_ratings = tv_shows.groupby('rating').size().reset_index(name='count')
    movies_ratings = movies.groupby('rating').size().reset_index(name='count')
    
    # Return the two DataFrames for TV Shows and Movies
    return tv_shows_ratings, movies_ratings

    
def get_ratings_distribution(data):
    # Calculate rating counts for TV Shows and Movies
    tv_rating_counts = data[data['type'] == 'TV Show']['rating'].value_counts()
    movie_rating_counts = data[data['type'] == 'Movie']['rating'].value_counts()

    # Combine into a DataFrame
    ratings_combined = pd.DataFrame({'TV Shows': tv_rating_counts, 'Movies': movie_rating_counts}).fillna(0)

    return ratings_combined  # Return the combined ratings for later use

def process_rating_counts(tv_rating_counts, movie_rating_counts):
    # You can process the counts further, combine them, or do analysis
    combined_counts = pd.DataFrame({
        'TV Shows': tv_rating_counts,
        'Movies': movie_rating_counts
    }).fillna(0)

    # You can return this DataFrame, or perform further processing here
    return combined_counts

def view_ratings_country_tv(data):
    # Filter the TV Shows and exclude invalid ratings
    tv_shows = data[(data['type'] == 'TV Show') & (~data['rating'].fillna('').str.contains('min'))]
    
    # Count ratings for each TV Show
    tv_rating_counts = tv_shows['rating'].value_counts()
    
    # Create DataFrame for TV Shows rating counts
    tv_rating_counts_df = pd.DataFrame(tv_rating_counts).reset_index()
    tv_rating_counts_df.columns = ['Rating', 'Count']
    
    # Plot the bar chart for TV Shows
    plt.figure(figsize=(10, 6))
    plt.bar(tv_rating_counts_df['Rating'], tv_rating_counts_df['Count'], color='skyblue')
    plt.title('Distribution of Ratings for TV Shows')
    plt.xlabel('Rating')
    plt.ylabel('Number of TV Shows')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def view_ratings_country_movies(data):
    # Filter the Movies and exclude invalid ratings
    movies = data[(data['type'] == 'Movie') & (~data['rating'].fillna('').str.contains('min'))]
    
    # Count ratings for each Movie
    movie_rating_counts = movies['rating'].value_counts()
    
    # Create DataFrame for Movie rating counts
    movie_rating_counts_df = pd.DataFrame(movie_rating_counts).reset_index()
    movie_rating_counts_df.columns = ['Rating', 'Count']
    
    # Plot the bar chart for Movies
    plt.figure(figsize=(10, 6))
    plt.bar(movie_rating_counts_df['Rating'], movie_rating_counts_df['Count'], color='lightcoral')
    plt.title('Distribution of Ratings for Movies')
    plt.xlabel('Rating')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Compute tv_rating_counts for TV Shows
tv_shows = data[(data['type'] == 'TV Show') & (~data['rating'].fillna('').str.contains('min'))]
tv_rating_counts = tv_shows['rating'].value_counts()
# Compute movie_rating_counts for Movies
movies = data[(data['type'] == 'Movie') & (~data['rating'].fillna('').str.contains('min'))]
movie_rating_counts = movies['rating'].value_counts()
# Now create ratings_combined DataFrame
ratings_combined = pd.DataFrame({'TV Shows': tv_rating_counts, 'Movies': movie_rating_counts})
# Convert 'rating' into numerical format
data['rating_numeric'] = LabelEncoder().fit_transform(data['rating'].astype(str))

# We’ll use only numerical features for clustering
X = data[['rating_numeric', 'release_year']].dropna()


def preprocess_data(data):
    # Add 'release_year' column
    if 'release_year' not in data.columns:
        if 'date_added' in data.columns:
            data['release_year'] = pd.to_datetime(data['date_added'], errors='coerce').dt.year
        else:
            raise ValueError("'date_added' column is missing; cannot extract 'release_year'.")

    # Add 'rating_numeric' column
    if 'rating_numeric' not in data.columns:
        rating_map = {
            'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5, 'TV-Y': 1, 'TV-Y7': 2,
            'TV-Y7-FV': 3, 'TV-G': 4, 'TV-PG': 5, 'TV-14': 6, 'TV-MA': 7
        }
        data['rating_numeric'] = data['rating'].map(rating_map).fillna(0)
    
    return data
def kmeans_plot(data):
    # Ensure you have the necessary columns for clustering
    if 'release_year' not in data.columns or 'rating_numeric' not in data.columns:
        st.error("Data does not have the required columns: 'release_year' and 'rating_numeric'.")
        return

    # Prepare the data for K-means clustering
    X = data[['release_year', 'rating_numeric']]

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)

    # Add a title with minimal space using inline CSS
    st.markdown(
        """
        <h2 style="margin-bottom: -40px; position:relative;  z-index:9999; padding-bottom: 0px;">K-means Clustering of Netflix Data</h2>
        """, 
        unsafe_allow_html=True
    )

    # Create an interactive plot with Plotly
    fig = px.scatter(
        data,
        x='release_year',
        y='rating_numeric',
        color='cluster',  # Color by cluster
        labels={'release_year': 'Release Year', 'rating_numeric': 'Rating (Numeric)', 'cluster': 'Cluster'},
        color_continuous_scale='viridis',
        size_max=8,  # Maximum size of the markers
        opacity=0.8,  # Set transparency for better visibility
        template='plotly_dark',  # Optional: Choose a different template for style
        width=1000,  # Width of the plot
        height=600,  # Height of the plot
    )

    # Adjust marker size for better visibility
    fig.update_traces(marker=dict(size=12))  # Increase marker size here

    # Show the interactive plot in Streamlit
    st.plotly_chart(fig)
    
def view_stacked_dist_ratings(data):
    # Create a DataFrame for stacking
    tv_rating_counts = data[data['type'] == 'TV Show']['rating'].value_counts()
    movie_rating_counts = data[data['type'] == 'Movie']['rating'].value_counts()
    
    ratings_combined = pd.DataFrame({
        'Rating': tv_rating_counts.index.union(movie_rating_counts.index), 
        'TV Shows': tv_rating_counts.reindex(tv_rating_counts.index.union(movie_rating_counts.index), fill_value=0),
        'Movies': movie_rating_counts.reindex(tv_rating_counts.index.union(movie_rating_counts.index), fill_value=0)
    }).reset_index(drop=True)
    st.markdown(
        """
        <h2 style="margin-bottom: -40px; position:relative;  z-index:9999; padding-bottom: 0px;">
        Distribution of Ratings for TV Shows and Movies</h2>
        """, 
        unsafe_allow_html=True
    )

    # Create an interactive stacked bar chart
    fig = px.bar(
        ratings_combined,
        x='Rating',
        y=['TV Shows', 'Movies'],
        labels={'value': 'Count', 'Rating': 'Rating'},
        color_discrete_sequence=['Skyblue', 'Lightgreen']
    )
    
    # Adjust layout for better aesthetics
    fig.update_layout(
        barmode='stack',  # Stacked bar chart mode
        width=1000,  # Chart width
        height=600,  # Chart height
        legend_title_text='Content Type'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def view_box_plot(data):
    # Display an H2 title
    st.markdown(
        """
        <h2 style="margin-bottom: -40px; position:relative;  z-index:9999; padding-bottom: 0px;">Interactive Box Plot of Ratings for Movies and TV Shows</h2>
        """, 
        unsafe_allow_html=True
    )

    # Encode ratings as numeric
    data['rating_numeric'] = LabelEncoder().fit_transform(data['rating'].astype(str))

    # Prepare data for Plotly
    plot_data = pd.DataFrame({
        'Rating (Numeric)': data['rating_numeric'],
        'Type': data['type']
    })

    # Create an interactive box plot
    fig = px.box(
        plot_data,
        x='Type',
        y='Rating (Numeric)',
        color='Type',
        labels={'Rating (Numeric)': 'Rating (Numeric)', 'Type': 'Content Type'},
        color_discrete_sequence=['Skyblue', 'Salmon']  # Custom colors for better aesthetics
    )

    # Adjust chart size
    fig.update_layout(
        width=900,  # Set the width of the chart
        height=600  # Set the height of the chart
    )

    # Display the box plot in Streamlit
    st.plotly_chart(fig, use_container_width=False)
# # Chart 5: Release Year Trends in Content Addition (Line Chart) - Kareem
# 
# * Track the trends in content addition over the years using machine learning time series analysis techniques.
# 

#using groupby on release year and type of video data
#restructure the data by unstacking so that inner index levels are turned columns
#chose to dropna() rather than fillna(0) because the latter would distort accuracy
def view_content_by_release_year(data):
    # Ensure the 'release_year' is formatted correctly (if needed)
    data['release_year'] = data['release_year'].astype(str).str[:4]  # Convert to 4-digit string (year format)
    
    # Group by both 'release_year' and 'type' (assuming 'type' column indicates 'movie' or 'tv_show')
    content_by_year = data.groupby(['release_year', 'type']).size().reset_index(name='count')
    
    return content_by_year

# # Chart 6: Content Duration by Type (Box Plot) - Kareem
# 
# * Analyse the distribution of content duration (in minutes) by type (TV show or movie)
# 
# 

# * Visualize the distribution using a box plot, where each box represents the distribution of duration for TV shows and movies separately.

# # Chart 7: Distribution of Content by Genre - Paige
# 
# * Aggregate TV shows and movies by genre
# 
# * Visualize the distribution using a stacked bar chart, where each bar represents a genre and the stacked segments represent the proportion of TV shows and movies within each genre.

def view_media_added_by_year(data):
    content_by_year = view_content_by_release_year(data)  # Get the data for content by year
    st.markdown(
    """
    <h2 style="margin-bottom: -40px; position:relative;  z-index:9999; padding-bottom: 0px;">Content Added by Year (Movies and TV Shows)</h2>
    """, 
    unsafe_allow_html=True
    )
    
    # Create an interactive Plotly line chart for both TV shows and Movies
    fig = px.line(
        content_by_year,
        x='release_year',  # X-axis is release year
        y='count',  # Y-axis is the count of content
        color='type',  # Differentiate between TV shows and Movies by color
        markers=True,  # Add markers for each point
        labels={'release_year': 'Release Year', 'count': 'Content Count', 'type': 'Content Type'},  # Labels
    )
    
    # Update the line and marker colors to the desired colors (Orange for TV Shows, Blue for Movies)
    fig.update_traces(
        line=dict(width=4),  # Make the line thicker
        marker=dict(size=10),  # Increase marker size
        selector=dict(name='TV Shows'),
        line_color='orange'  # Set TV Shows line to orange
    )
    
    fig.update_traces(
        line=dict(width=4),  # Make the line thicker
        marker=dict(size=10),  # Increase marker size
        selector=dict(name='Movies'),
        line_color='blue'  # Set Movies line to blue
    )
    
    # Display the interactive Plotly chart in Streamlit
    st.plotly_chart(fig)

##############################################################
########### ADD THIS CHART BY PAIGE INTO THE DASHBOARD ###########
##############################################################
data_exploded = data.copy()  # Create a copy of the original data
# split genres into lists
data_exploded['genres'] = data_exploded['listed_in'].str.split(', ')
data_exploded = data_exploded.explode('genres')  # explode lists into seperate rows

# get unique genres and do tf-idf vectorization over genres
unique_genres = data['listed_in'].str.split(', ').explode().unique()
genre_vec = TfidfVectorizer().fit_transform(unique_genres)

# apply PCA dimmensionality reduction
pca = PCA(n_components=15, random_state=21)  # reduce to 15 based on elbow
genre_vec_pca = pca.fit_transform(genre_vec.toarray())

#  GMM for soft clustering so the movies/tv shows can belong to more than one class
n_clusters = 14  # optimal cluster model based on silhoette score defined below
gmm = GaussianMixture(n_components=n_clusters, random_state=21).fit(genre_vec_pca)

# predict cluster
gmm_labels = gmm.predict(genre_vec_pca)
cluster_mapping = pd.DataFrame({'genre': unique_genres, 'cluster': gmm_labels})

#######_____<- This is heavily aided with AI idk how to cite *properly* but I can also rewrite just need help
#  clean cluster labels by taking out redundant info ("tv" or "movie")
def clean_cluster_label(label):
    """remove 'TV' or 'Movies' from genre names unless they are the only label."""
    if label.strip().lower() in ['movies', 'tv shows','series']:
      return label.strip()
    if label.strip().lower() == 'docuseries':
      return 'Documentary Style' # since docuseries is only for TV
    return re.sub(r'\b(TV|Movies|Series|Docuseries?)\b', '', label, flags=re.IGNORECASE).strip()

# label the clusters with meaningful representations
def get_cluster_label(cluster_num):
    """Find the most representative genre from genre list in each cluster based
    on cosine similarity."""
    # select a single PCA reduced cluster
    genres_in_cluster = cluster_mapping[cluster_mapping['cluster'] == cluster_num]['genre']
    genre_indices = [unique_genres.tolist().index(genre) for genre in genres_in_cluster]
    genre_vecs_in_cluster = genre_vec_pca[genre_indices]

    # calculate cosine similarities between center of cluster and vectors in it
    cluster_center = gmm.means_[cluster_num].reshape(1, -1)  # middle vector
    similarities = cosine_similarity(genre_vecs_in_cluster, cluster_center)

    # find the closest/most similar vector to label the cluster after
    most_representative_idx = similarities.argmax()
    representative_genre = genres_in_cluster.iloc[most_representative_idx]

    return clean_cluster_label(representative_genre)

# __________________________________________________________________


# give labels from above to clusters
cluster_labels = pd.Series(
    [get_cluster_label(i) for i in range(n_clusters)],
    index=range(n_clusters)
).to_dict()

# genre mapping
def map_to_cluster_label(genre):
    cluster = cluster_mapping[cluster_mapping['genre'] == genre]['cluster']
    if not cluster.empty:
        return cluster_labels[cluster.iloc[0]]
    else:
        return 'No Genre'

data_exploded['cluster_label'] = data_exploded['genres'].apply(map_to_cluster_label)

# use Groupby for label and type then sum # in each group
cluster_type_dist = (
    data_exploded.groupby(['cluster_label', 'type'])
    .size().reset_index(name='counts'))

# plot
fig = px.bar(
    cluster_type_dist,
    x='cluster_label',
    y='counts',
    color="type",
    title='Distribution of Content by Genre',
    labels={'counts': 'Number of Titles',
            'cluster_label': 'Clustered Genres'},
    barmode='stack')

# make plot pretty
fig.update_layout(
    xaxis={'categoryorder': 'total descending'},
    xaxis_tickangle=-50)

fig.show()

# so this data is exploded - so counted multiple times in the visualization
# (ie we only have x number of movies but if each movie was assigned 3 genres 3x
# titles appear on the graph)



##############################################################
###### ADJUSTED CODE FOR TEH DASHBOARD #########################
def display_genre_distribution(data):
    data_exploded = data.copy()  # Create a copy of the original data
    # Split genres into lists
    data_exploded['genres'] = data_exploded['listed_in'].str.split(', ')
    data_exploded = data_exploded.explode('genres')  # Explode lists into separate rows

    # Get unique genres and do TF-IDF vectorization over genres
    unique_genres = data['listed_in'].str.split(', ').explode().unique()
    genre_vec = TfidfVectorizer().fit_transform(unique_genres)

    # Apply PCA dimensionality reduction
    pca = PCA(n_components=15, random_state=21)  # Reduce to 15 based on elbow
    genre_vec_pca = pca.fit_transform(genre_vec.toarray())

    # GMM for soft clustering so the movies/TV shows can belong to more than one class
    n_clusters = 14  # Optimal cluster model based on silhouette score
    gmm = GaussianMixture(n_components=n_clusters, random_state=21).fit(genre_vec_pca)

    # Predict cluster
    gmm_labels = gmm.predict(genre_vec_pca)
    cluster_mapping = pd.DataFrame({'genre': unique_genres, 'cluster': gmm_labels})

    # Clean cluster labels by removing redundant info
    def clean_cluster_label(label):
        """Remove 'TV' or 'Movies' from genre names unless they are the only label."""
        if label.strip().lower() in ['movies', 'tv shows', 'series']:
            return label.strip()
        if label.strip().lower() == 'docuseries':
            return 'Documentary Style'  # Since docuseries is only for TV
        return re.sub(r'\b(TV|Movies|Series|Docuseries?)\b', '', label, flags=re.IGNORECASE).strip()

    # Label the clusters with meaningful representations
    def get_cluster_label(cluster_num):
        """Find the most representative genre from genre list in each cluster based on cosine similarity."""
        # Select a single PCA reduced cluster
        genres_in_cluster = cluster_mapping[cluster_mapping['cluster'] == cluster_num]['genre']
        genre_indices = [unique_genres.tolist().index(genre) for genre in genres_in_cluster]
        genre_vecs_in_cluster = genre_vec_pca[genre_indices]

        # Calculate cosine similarities between center of cluster and vectors in it
        cluster_center = gmm.means_[cluster_num].reshape(1, -1)  # Middle vector
        similarities = cosine_similarity(genre_vecs_in_cluster, cluster_center)

        # Find the closest/most similar vector to label the cluster
        most_representative_idx = similarities.argmax()
        representative_genre = genres_in_cluster.iloc[most_representative_idx]

        return clean_cluster_label(representative_genre)

    # Give labels from above to clusters
    cluster_labels = pd.Series(
        [get_cluster_label(i) for i in range(n_clusters)],
        index=range(n_clusters)
    ).to_dict()

    # Genre mapping
    def map_to_cluster_label(genre):
        cluster = cluster_mapping[cluster_mapping['genre'] == genre]['cluster']
        if not cluster.empty:
            return cluster_labels[cluster.iloc[0]]
        else:
            return 'No Genre'

    data_exploded['cluster_label'] = data_exploded['genres'].apply(map_to_cluster_label)

    # Use Groupby for label and type then sum in each group
    cluster_type_dist = (
        data_exploded.groupby(['cluster_label', 'type'])
        .size().reset_index(name='counts')
    )
      # Plot
    fig = px.bar(
        cluster_type_dist,
        x='cluster_label',
        y='counts',
        color="type",
        title='Distribution of Content by Genre',
        labels={'counts': 'Number of Titles',
                'cluster_label': 'Clustered Genres'},
        barmode='stack'
    )

    # Make plot pretty
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        xaxis_tickangle=-50
    )

    # Show the plot
    st.plotly_chart(fig)



###############################################################
###############################################################


# Code I used for checking things and making visualization but
# dont need in actual model - We should drop this but if anyone wants to check for logic probs

from sklearn.metrics import silhouette_score

# loop to find best component number
for n in range(5, 30):
    gmm = GaussianMixture(n_components=n, random_state=21).fit(genre_vec_pca)
    gmm_labels = gmm.predict(genre_vec_pca)
    sil_score = silhouette_score(genre_vec_pca, gmm_labels)
    print(f"GMM Silhouette Score with {n} components: {sil_score:.3f}")

## code to check if its working well enough

title_to_inspect = data_exploded['title'].sample(n=180).iloc[0]

# Get the data for that title
item_data = data_exploded[data_exploded['title'] == title_to_inspect]

print(f"\nInspecting item: {title_to_inspect}")
if item_data.empty:
    print(f"No data found for title: {title_to_inspect}")
else:
    # Get the original genres from the 'data' DataFrame
    original_genres = data[data['title'] == title_to_inspect]['listed_in'].iloc[0]
    print(f"Original genres: {original_genres}")
    print("\nAssigned cluster labels:")
    print(item_data[['title', 'type', 'genres', 'cluster_label']])


# # Chart 8: Content Addition by Date
# 
# * Analyse the pattern of content addition by date (month and year) using ML technique
# 
# * Visualize the pattern using a heatmap, where each cell represents the number of TV shows and movies added on a specific date.

#As it changes datetime format for all, so it doesnt try reformatting after it changes
try:
  data['date_added'] = pd.to_datetime(data['date_added'].str.strip(), format='%B %d, %Y')
except:
  print("Date is already formatted")
#Making year and month columns for making the heatmap
data['year'] = data['date_added'].dt.year
data['month'] = data['date_added'].dt.month

#Grouping by year and month, counting occurunces
content_by_date = data.groupby(['year', 'month']).size().reset_index(name='count')

#Pivoting for the heatmap
content_pivot = content_by_date.pivot(index='year', columns='month', values='count')

#Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(content_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5)
plt.title('Content Addition by Date')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

# Inside team_code.py

def generate_genre_distribution_plot(data):
    import matplotlib.pyplot as plt
    genre_counts = data['genre'].value_counts()
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar')
    plt.title('Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.show()

## ADDING MOHAMED CODE HERE
# Helper function to preprocess data
def preprocess_data_mohamed(data):
    # Clean and transform data
    data_dropna = data.dropna().reset_index()
    data_genre = data_dropna.copy()
    data_genre['genre'] = data_genre['listed_in'].apply(lambda x: [item.strip() for item in x.split(',')])
    data_genre = data_genre.explode('genre')

    def get_duration_val(duration):
        return int(duration.split()[0])

    data_genre['new_duration'] = data_genre['duration'].apply(get_duration_val)

    # Separate movies and calculate genre statistics
    movies_list = data_genre[data_genre['type'] == 'Movie']
    duration_genre = movies_list.groupby('genre').agg(
        average_duration=('new_duration', 'mean'),
        count=('new_duration', 'count')
    ).reset_index().sort_values(by='average_duration')

    return data_genre, movies_list, duration_genre


# Helper function to normalize movie data
def normalize_movies_data(movies_list):
    unique_genres = movies_list['genre'].unique()
    genre_dict = {genre: i for i, genre in enumerate(unique_genres)}
    movies_list['genre_code'] = movies_list['genre'].map(genre_dict)

    # Calculate normalization metrics
    mean_release_year = movies_list['release_year'].mean()
    std_release_year = movies_list['release_year'].std()
    mean_duration = movies_list['new_duration'].mean()
    std_duration = movies_list['new_duration'].std()
    mean_genre = movies_list['genre_code'].mean()
    std_genre = movies_list['genre_code'].std()

    # Normalize data
    movies_list_normalized = movies_list.copy().reset_index()
    movies_list_normalized['release_year'] = (movies_list_normalized['release_year'] - mean_release_year) / std_release_year
    movies_list_normalized['new_duration'] = (movies_list_normalized['new_duration'] - mean_duration) / std_duration
    movies_list_normalized['genre_code'] = (movies_list_normalized['genre_code'] - mean_genre) / std_genre

    return movies_list_normalized, mean_duration, std_duration


# Train and evaluate linear regression model
def train_model(movies_list_normalized):
    X = movies_list_normalized[['release_year', 'genre_code']]
    y = movies_list_normalized['new_duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return model, mse


# Predict movie durations
def predict_durations(model, movies_list, mean_duration, std_duration):
    predictions = (model.predict(movies_list[['release_year', 'genre_code']]) * std_duration) + mean_duration
    movies_list['predicted_duration'] = predictions
    return movies_list


# Generate country-based aggregated data
def aggregate_by_country(data, content_type):
    content_data = data[data['type'] == content_type]
    grp_data = content_data.groupby('country').agg({'title': 'count'}).reset_index()
    grp_data.columns = ['Country', f'Total {content_type}s']
    threshold = 10 if content_type == 'Movie' else 5
    filtered_data = grp_data[grp_data[f'Total {content_type}s'] > threshold].sort_values(by=f'Total {content_type}s')
    return filtered_data


def heatmap_chart(data):
    #As it changes datetime format for all, so it doesnt try reformatting after it changes
    try:
        data['date_added'] = pd.to_datetime(data['date_added'].str.strip(), format='%B %d, %Y')
    except:
        print("Date is already formatted")
    #Making year and month columns for making the heatmap
    data['year'] = data['date_added'].dt.year
    data['month'] = data['date_added'].dt.month

    #Grouping by year and month, counting occurunces
    content_by_date = data.groupby(['year', 'month']).size().reset_index(name='count')

    #Pivoting for the heatmap
    content_pivot = content_by_date.pivot(index='year', columns='month', values='count')
    heatmap = alt.Chart(content_by_date).mark_rect().encode(
    x='month:O',
    y='year:O',
    color='count:Q'
    )
    st.altair_chart(heatmap, use_container_width=True)
