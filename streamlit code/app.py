import os

import streamlit as st
import altair as alt
import pandas as pd
import streamlit.components.v1 as components

# Import functions from the team_code.py file #####################
from team_code import (
    load_data, 
    view_our_data, 
    view_our_data_info, 
    view_our_data_columns, 
    view_our_data_description,
    view_tv_vs_movie,
    view_tv_movie_rating_distribution,
    view_tv_country,
    view_movie_country,
    view_movie_ratings_by_country,
    view_tv_show_ratings_by_country,
    kmeans_plot,
    preprocess_data,
    view_stacked_dist_ratings,
    view_box_plot,
    view_content_by_release_year,
    view_media_added_by_year,
    normalize_movies_data, 
    train_model, 
    predict_durations, 
    aggregate_by_country,
    preprocess_data_mohamed,
    generate_genre_distribution_plot,
    display_genre_distribution,
    page_Chart_1, 
    page_Chart_2,
    page_Chart_3,
    page_Chart_4,
    heatmap_chart

)
##############################################################
# Load data
data = load_data('netflix_titles.csv')
data = preprocess_data(data)

# Custom CSS for padding and vertical alignment
st.markdown("""
    <style>
        .main-content {
            margin: 0 50px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to display content inside main-content div
def render_content(content_func):
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    content_func()
    st.markdown("</div>", unsafe_allow_html=True)

# Page Content Functions
def introduction():
    st.image('streamlit code/images/fanLogo.png') 
    st.markdown("""
    <h1 style="color:red;">Capstone Project</h1>
    <h2>Netflix Dataset Analysis</h2>
    <p>This project is related to data visualization and analysis of Netflix movies and TV shows.</p>
    <h3><strong>INFO 6151: Data Visualization</strong></h3>
    <hr>
    <p>You can find the <a href="https://github.com/m-hossni/data_visualization_ML_F24" target="_blank">GitHub repository here</a>.</p>
    """, unsafe_allow_html=True)

###### FUCNTIONS FOR PLOTS AND DATA DISPLAY ##################
def data_table():
    st.write("### Netflix TV Shows and Movies Dataset")
    st.write("This dataset contains information about TV Shows and Movies available on Netflix.")
    st.write("The head() of the dataset is shown below:")  
    view_our_data(data)

def data_info():
    view_our_data_info(data)

def data_description():
    view_our_data_description(data)

def data_columns():
    view_our_data_columns(data)

def data_tv_movie():
    # View TV Shows vs Movies distribution
    view_tv_vs_movie(data)

def data_movie_country():
    # Get tv_shows and movies
    tv_shows, movies = view_tv_movie_rating_distribution(data)
    
    # Plot the interactive distribution plot for Movies
    #plot_movie_distribution(movies)
    view_movie_country(data)

def data_tv_country():
    # Get tv_shows and movies
    tv_shows, movies = view_tv_movie_rating_distribution(data)
    view_tv_country(data)
    
    # Plot the interactive distribution plot for TV Shows
    #plot_tv_distribution(tv_shows)

def ratings_movie_country():
    # Get tv_shows and movies
    tv_shows, movies = view_tv_movie_rating_distribution(data)
    #view_movie_country(data)
    #view_tv_movie_rating_distribution(data)
    view_movie_ratings_by_country(data)
    
def stacked_bars():
    view_stacked_dist_ratings(data)
    
    
def ratings_tv_country():
    # Get tv_shows and movies
    tv_shows, movies = view_tv_movie_rating_distribution(data)
    
    # Only show TV Ratings plot when selected
    #view_tv_movie_rating_distribution(data)
    view_tv_show_ratings_by_country(data)
    #plot_tv_ratings(tv_shows)


def charts_kmeans():
    kmeans_plot(data)

def charts_box():
    view_box_plot(data)

def content_year_releases(data):
    # Get the content_by_year DataFrame from team_code.py
    content_by_year = view_content_by_release_year(data)
    
    # Display the content_by_year DataFrame as a table
    st.subheader("Content Distribution by Release Year (Table)")
    st.dataframe(content_by_year, use_container_width=True)
    
def release_years():  # Pass data here as a parameter
    content_year_releases(data)
    
def line_added_media():
    view_media_added_by_year(data)

def content_genre():
    # Generate and display the plot
    fig = generate_genre_distribution_plot(data)
    st.plotly_chart(fig, use_container_width=True)


def view_references_citations():
    st.image('streamlit_code/images/fanLogo.png') 
    st.title("References and AI Assistance Disclosure")
    
    st.markdown(
        """
        ### References
        
        **Gaussian Mixture Models (GMM):**  
        Scikit-learn Developers. (n.d.). *GaussianMixture model*. In *Scikit-learn 1.3.1 documentation*. Retrieved October 23, 2024, from  
        [https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
        
        **Principal Component Analysis (PCA):**  
        Scikit-learn Developers. (n.d.). *PCA model*. In *Scikit-learn 1.3.1 documentation*. Retrieved October 25, 2024, from  
        [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
        
        **AI Assistance:**  
        OpenAI. (2024). *ChatGPT* (Version 4). OpenAI. Retrieved from  
        [https://openai.com](https://openai.com).
        
        ---
        
        ### AI Assistance Disclosure
        
        Some parts of this project were developed with the assistance of an AI model, *ChatGPT by OpenAI*. This included:
        
        - Guidance on clustering techniques such as K-means and Gaussian Mixture Models (GMM).
        - Recommendations for improving visualizations (e.g., line charts, box plots, and heatmaps).
        - Suggestions for structuring and refining code logic.
        
        While the implementation and final execution were carried out independently, the AI contributed by enhancing the clarity and robustness of the methods applied.
        """,
        unsafe_allow_html=True
    )
    
def get_genre_data():
    # Call the function from team_code.py to display the genre distribution and data
    display_genre_distribution(data)
    
##########################################################

    
#### MOHAMED CODE TO GO HERE ##############################
##########################################################
# Preprocess and train
data_genre, movies_list, duration_genre = preprocess_data_mohamed(data)
movies_list_normalized, mean_duration, std_duration = normalize_movies_data(movies_list)
model, mse = train_model(movies_list_normalized)
movies_list = predict_durations(model, movies_list, mean_duration, std_duration)

# Define pages
def page_chart_2_1():
    st.title(" Average Movie Duration by Genre")
    st.bar_chart(data=duration_genre, x="genre", y="average_duration", use_container_width=True)
# JavaScript snippet to get viewport dimensions
# JavaScript snippet to get viewport dimensions
def get_viewport_dimensions():
    viewport_script = """
    <script>
    const rect = document.body.getBoundingClientRect();
    document.body.dataset.width = rect.width;
    document.body.dataset.height = rect.height;
    </script>
    """
    components.html(viewport_script, height=0)

    # Retrieve the dimensions via query parameters
    viewport_width = st.query_params.get("width", [800])[0]  # Updated here
    viewport_height = st.query_params.get("height", [600])[0]  # Updated here

    return int(viewport_width), int(viewport_height)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data_dropna = data.dropna().reset_index()
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


data_genre = data_dropna.copy()
data_genre['genre'] = data_genre['listed_in'].apply(lambda x : [item.strip() for item in x.split(',')])
data_genre = data_genre.explode('genre')
#data_genre


def get_duration_val(duration):
  return int(duration.split()[0])
data_genre['new_duration'] = data_genre['duration'].apply(get_duration_val)
#data_genre

movies_list = data_genre[data_genre['type'] == 'Movie']
duration_genre = movies_list.groupby('genre').agg(
    average_duration =('new_duration', 'mean'),
    count = ('new_duration', 'count')
).reset_index().sort_values(by='average_duration')


unique_genres = movies_list['genre'].unique()
num_genre = len(unique_genres)



#converting categorical data into numbers
movies_transformed = movies_list.copy()
genre_dict = {}
for i, genre in enumerate(unique_genres):
  genre_dict[genre] = i


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

predictions = (model.predict(X)*std_duration) +mean_duration
movies_list['predicted_duration'] = predictions
def page_chart_2_2():
    st.title("Chart 2.2: Release Year vs Duration (Prediction)")
    st.write(f"The Mean Squared Error from the model is: {mse:.4f}")

    option = st.selectbox("Select Genre", movies_list["genre"].unique())

    # Filter data based on selected genre
    genre_data = movies_list[movies_list['genre'] == option]

    if genre_data.empty:
        st.write("No data available for this genre.")
        return

    # Define scales (manually adjust the range if necessary)
    x_scale = alt.Scale(domain=[genre_data["release_year"].min(), genre_data["release_year"].max()])
    y_scale = alt.Scale(domain=[genre_data["new_duration"].min(), genre_data["new_duration"].max()])

    # Scatter plot and line plot
    base = alt.Chart(genre_data).mark_circle(size=80).encode(
        x=alt.X("release_year", scale=x_scale, title="Release Year"),
        y=alt.Y("new_duration", scale=y_scale, title="Duration (minutes)"),
        tooltip=["title", "release_year", "new_duration"]
    )

    line = alt.Chart(genre_data).mark_line(color="red", strokeWidth=2).encode(
        x=alt.X("release_year"),
        y=alt.Y("predicted_duration")
    )

    # Combine scatter plot and line
    chart = base +line
    #chart = chart.properties(
    #    width=800,  # Customize chart width
    #    height=400  # Customize chart height
    #)

    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
def page_chart_3():
    st.title("Content Distribution by Country")
    st.write("### Movies")
    filtered_movies = aggregate_by_country(data, "Movie")
    movie_chart = alt.Chart(filtered_movies).mark_bar().encode(
        x="Country",
        y="Total Movies",
        tooltip=["Country", "Total Movies"]
    )
    st.altair_chart(movie_chart, use_container_width=True)

    st.write("### TV Shows")
    filtered_shows = aggregate_by_country(data, "TV Show")
    tv_chart = alt.Chart(filtered_shows).mark_bar().encode(
        x="Country",
        y="Total TV Shows",
        tooltip=["Country", "Total TV Shows"]
    )
    st.altair_chart(tv_chart, use_container_width=True)

def heatmap():
    st.write("Heatmap for movie release per year and month")
    heatmap_chart(data)

###########################################################
###########################################################

# Navigation
pages = {
    "Average Movie Duration": page_chart_2_1,
    "Release Year vs Duration": page_chart_2_2,
    "Distribution by Country": page_chart_3
}
# Sidebar Navigation
def sidebar_navigation():
     # Add an image above the page selection dropdown
    st.sidebar.image("streamlit code/images/GroupLogo.png", use_container_width=True)
    menu = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Introduction",
            "Data Prep and Preprocessing",
            "\u2003\u2003\u2003🧮 Data Table",
            "\u2003\u2003\u2003🧮 Data Info",
            "\u2003\u2003\u2003🧮 Data Description",
            "\u2003\u2003\u2003🧮 Data Columns",
            "📊 TV Shows vs Movies",
            "\u2003\u2003\u2003🥧 Distribution of TV Shows vs Movies",
            "\u2003\u2003\u2003📊 Distribution of Content by Genre",
            #"📊 Relationship between Release Year and Duration (Bar Chart)",
            #"📊 Relationship between Release Year and Duration (Regression)",
            "📊 Average Movie Duration",  # Added MOHAMED's Chart 2.1
            "📊 Release Year vs Duration",  # Added MOHAMED's Chart 2.2
            "📊 Distribution by Country",  # Added MOHAMED's Chart 3
            #"Movie and TV Show Distribution by Country",
            #"\u2003\u2003\u2003🌍 Distribution of TV Shows by Country",
            #"\u2003\u2003\u2003🌍 Distribution of Movies by Country",
            "Movie and TV Show Ratings",
            "\u2003\u2003\u2003⭐ Movie Ratings by Country",
            "\u2003\u2003\u2003⭐ TV Show Ratings by Country",
            "📈 Distribution of Content by Rating",
            "\u2003\u2003\u2003🧮 K-Means Clustering Content Distribution",
            "\u2003\u2003\u2003📊 Stacked Bar Chart Distribution of Ratings",
            "\u2003\u2003\u2003📈 Box Plot Distribution of Ratings",
            "🎥 Release Year Trends",
            "\u2003\u2003\u2003⚙️ Content Addition by Year",
            "\u2003\u2003\u2003🛠️ Number of Movies or TV Shows Added Annually",
            "\u2003\u2003\u2003🛠️ Number of Movies Released Heatmap",
            "✅ References"
        ]
    )
   
   # Add a heading (h1) and paragraph with HTML content below the dropdown menu, centered
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 24px; color: #333;">Group 3</h1>
            <p style="font-size: 14px; color: #333;">
                Jack Ivanisevic, <br>Mohamed Ali Hossni, <br>Paige Berrigan, <br>Kareem Idris, <br>Cantrin Chapman 
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if menu in ["Data Prep and Preprocessing", 
                "Movie and TV Show Distribution by Country", 
                "Movie and TV Show Ratings", 
                "📈 Distribution of Content by Rating",
                "🎥 Release Year Trends",
                "📊 TV Shows vs Movies"]:
        st.sidebar.markdown("This option is not selectable.")
        st.warning("Please choose another option.")
        return None
    return menu

# Main Layout
menu = sidebar_navigation()
if menu:
    render_content({
        "🏠 Introduction": introduction,
        "\u2003\u2003\u2003🧮 Data Table": data_table,
        "\u2003\u2003\u2003🧮 Data Info": data_info,
        "\u2003\u2003\u2003🧮 Data Description": data_description,
        "\u2003\u2003\u2003🧮 Data Columns": data_columns,
        "\u2003\u2003\u2003🥧 Distribution of TV Shows vs Movies": data_tv_movie,
        "\u2003\u2003\u2003📊 Distribution of Content by Genre": get_genre_data, 
        "\u2003\u2003\u2003🌍 Distribution of TV Shows by Country": data_tv_country,
        "\u2003\u2003\u2003🌍 Distribution of Movies by Country": data_movie_country,
        "\u2003\u2003\u2003⭐ Movie Ratings by Country": ratings_movie_country,
        "\u2003\u2003\u2003⭐ TV Show Ratings by Country": ratings_tv_country,
        "\u2003\u2003\u2003🧮 K-Means Clustering Content Distribution": charts_kmeans,
        "\u2003\u2003\u2003📊 Stacked Bar Chart Distribution of Ratings": stacked_bars,
        "\u2003\u2003\u2003📈 Box Plot Distribution of Ratings": charts_box,
        "\u2003\u2003\u2003⚙️ Content Addition by Year": release_years,
        "\u2003\u2003\u2003🛠️ Number of Movies or TV Shows Added Annually": line_added_media,
        "📊 Average Movie Duration": page_chart_2_1,  # Added MOHAMED's Chart 2.1
        "📊 Release Year vs Duration": page_chart_2_2,  # Added MOHAMED's Chart 2.2
        "📊 Distribution by Country": page_chart_3,  # Added MOHAMED's Chart 3
        "\u2003\u2003\u2003🛠️ Number of Movies Released Heatmap": heatmap,
        "✅ References": view_references_citations,
    }.get(menu, lambda: st.write("Page not found")))
