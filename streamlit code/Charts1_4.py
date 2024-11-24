"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


###############################################################################
############################ introduction section #############################
################### Reading data and showing description ######################
###############################################################################
st.image('./images/logo.jpg',width=300)
st.logo('./images/logo.jpg')
st.html("<p style='font-size:12px; padding:0; margin:0; color:grey'>fanshawe college of information technology<br>artifical intelligence and machine learning</p>")
def page_introduction():
   #st.title("Data Visualization Project")
   #st.title("Group 3")
   #st.write("This project is related to data visualization and analysis of netflix movies and TV shows dataset")
   html_text = "<div>\
                <div style='margin:10px; margin-top:30px'>\
                    <p style='margin:0;'>INFO 6151 | Data Visualization<p>\
                    <h1 style='color:red;margin:0;padding:0'>Capstone Project</h1>\
                    <p>Netflix Dataset Analysis</p>\
                </div>\
                </div>\
                <a href='https://github.com/m-hossni/data_visualization_ML_F24'>The project Repo is availble here</a>"
   st.html(html_text)
data = pd.read_csv('netflix_titles.csv')
def page1():
    st.title("Netflex TV Shows and Movies Dataset")
    data

def page2():
   st.write("Data Description for Netflix TV shows and Movies")
   describe = data.describe(include='all')
   describe






###############################################################################
############################# for chart 1 Paige ###############################
########### show the movie v tv distrubution and their value counts ###########
###############################################################################

tv_mov_dist = pd.DataFrame(data['type'].value_counts())
tv_mov_dist= tv_mov_dist.reset_index()

def page_Chart_1():
   #tv_mov_dist

   st.write("# Chart 1: Distribution of TV Shows and Movies by Type")
   st.write("*   Use ML algorithm to analyse the distribution of TV shows and movies.")
   st.write("* Visualize the proportion using a pie chart, where each slice represents the percentage of TV shows and movies.")
   base = alt.Chart(tv_mov_dist).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field='count', type="quantitative", stack=True),
    color=alt.Color(field='type', type="nominal"),
    ).transform_joinaggregate(
    TotalCount='sum(count)',
    ).transform_calculate(
    PercentOfTotal="datum.count / datum.TotalCount"
    )
   pie = base.mark_arc(innerRadius=80)
   text = base.mark_text(radius=200,fill= "black", size=20).encode(alt.Text(field='PercentOfTotal', type="quantitative", format=".0%")) 
   donut_chart = pie + text
   st.altair_chart(donut_chart, use_container_width=True)


###############################################################################
############################# for chart 2 Mohamed #############################
###############################################################################

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

def page_Chart_2_1():
    st.write("# Chart 2: Relationship between Release Year and Duration")
    st.write("* Perform predictive analysis to understand the relationship between release year and duration of TV shows or movies using machine learning regression techniques.")
    st.write("* Develop a predictive model to estimate the duration based on the release year.")
    st.write("* Create a scatter plot visualization to depict the relationship between release year and duration.")
    st.bar_chart(data=duration_genre,
            x='genre',
            y='average_duration',
            x_label='Movie genre',
            y_label='Duration (min)',
            color=None, 
            horizontal=False, 
            stack=None, 
            width=None, 
            height=None, 
            use_container_width=True)  

def page_Chart_2_2():
    st.write("# Chart 2: Relationship between Release Year and Duration")
    st.write("* Perform predictive analysis to understand the relationship between release year and duration of TV shows or movies using machine learning regression techniques.")
    st.write("* Develop a predictive model to estimate the duration based on the release year.")
    st.write("* Create a scatter plot visualization to depict the relationship between release year and duration.")
    st.write("The Mean Squared Error from regression is: {:.4f}".format(mse))
    option = st.selectbox('Select genre', unique_genres, index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")

    base = alt.Chart(movies_list[movies_list['genre']== option]).encode(x=alt.X('release_year', scale=alt.Scale(domain=[movies_list['release_year'].min(), movies_list['release_year'].max()]))).mark_circle().encode(y='new_duration').interactive()
    #bar = base.mark_circle()
    line = base.mark_line(color='red').encode(y='predicted_duration')
    final = base + line
    st.altair_chart(final, use_container_width=True)



###############################################################################
############################# for chart 3 Mohamed #############################
###############################################################################


data_shows = data[data['type'] == 'TV Show']
grp_data = data_shows.groupby('country').agg({
    'title':'count'
}).reset_index()
grp_data.columns = ['Country', 'Total Shows']
filtered_data = grp_data[grp_data['Total Shows'] > 5].sort_values(by='Total Shows')


data_Movies = data[data['type'] == 'Movie']
grp_data_movies = data_Movies.groupby('country').agg({
    'title':'count'
}).reset_index()
grp_data_movies.columns = ['Country', 'Total Movies']
filtered_data = grp_data_movies[grp_data_movies['Total Movies'] > 10].sort_values(by='Total Movies')



def page_Chart_3():
    st.write("# Chart 3: Distribution of Content by Country")
    st.write("* Aggregate TV shows and movies by country using ML technique.")
    st.write("* Visualize the distribution using a bar chart, where each bar represents a country and the height represents the number of TV shows or movies produced.")
    bar_plot_movies = alt.Chart(data=filtered_data).encode(x=alt.X('Country').sort('-y')).mark_bar().encode(y='Total Movies').interactive()
    st.altair_chart(bar_plot_movies, use_container_width=True)

###############################################################################
############################### Chart 4 work ##################################
####################### Two Pages combining Jack's work #######################
###############################################################################


tv_shows = data[(data['type'] == 'TV Show') & (~data['rating'].fillna('').str.contains('min'))]
movies = data[(data['type'] == 'Movie') & (~data['rating'].fillna('').str.contains('min'))]
tv_rating_counts = tv_shows['rating'].value_counts()
movie_rating_counts = movies['rating'].value_counts()
tv_rating_counts_df = pd.DataFrame(tv_rating_counts).reset_index()
tv_rating_counts_df.columns = ['Rating', 'Count']
movie_rating_counts_df = pd.DataFrame(movie_rating_counts).reset_index()
movie_rating_counts_df.columns = ['Rating', 'Count']
def page_Chart_4_1():
   st.write("# Chart 4: Distribution of Content by Rating")
   st.write("* Analyse the distribution of TV shows and movies by rating using ML clustering techniques.")
   st.write("* Visualize the distribution using a bar chart, where each bar represents a rating category (e.g., G, PG, PG-13, R, etc.) and the height represents the number of TV shows or movies in each category.")

   option = st.selectbox('Select type', ['Movies', 'TV Shows'], index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
   st.bar_chart(
      data = tv_rating_counts_df if option == 'TV Shows' else movie_rating_counts_df,
      x= 'Rating',
      y= 'Count',
      x_label='Rating',
      y_label='Number of TV Shows',
      color=None, 
      horizontal=False, 
      stack=None, 
      width=None, 
      height=None, 
      use_container_width=True
    ) 



# Convert 'rating' into numerical format
data['rating_numeric'] = LabelEncoder().fit_transform(data['rating'].astype(str))

# We’ll use only numerical features for clustering
X = data[['rating_numeric', 'release_year']].dropna()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
X['cluster'] = kmeans.fit_predict(X)
ratings_combined_tv = pd.DataFrame({'count': tv_rating_counts, 'type': len(tv_rating_counts)*['TV Shows']}).fillna(0).reset_index()
ratings_combined_movies = pd.DataFrame({'count': movie_rating_counts, 'type': len(movie_rating_counts)*['Movies']}).fillna(0).reset_index()        
ratings_combined = pd.concat([ratings_combined_movies, ratings_combined_tv], axis=0)


def page_chart_4_2():
    #ratings_combined_tv
    #ratings_combined_movies
    #ratings_combined
    st.write("# Chart 4: Distribution of Content by Rating")
    st.write("* Analyse the distribution of TV shows and movies by rating using ML clustering techniques.")
    st.write("* Visualize the distribution using a bar chart, where each bar represents a rating category (e.g., G, PG, PG-13, R, etc.) and the height represents the number of TV shows or movies in each category.")
    option = st.selectbox('Select plot', ['Clustered Data', 'Movies and TV Shows Clustered Data', 'Box Plot'], index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
    if option == 'Clustered Data':
       base = alt.Chart(X).encode(x=alt.X('release_year', scale=alt.Scale(domain=[X['release_year'].min(), X['release_year'].max()]))).mark_circle().encode(y='rating_numeric', color='cluster:N').interactive() 
    elif option == 'Movies and TV Shows Clustered Data':
       base = alt.Chart(ratings_combined).mark_bar().encode(x='rating', y='count', color='type')
    else:
       base = alt.Chart(data).mark_boxplot(extent="min-max").encode(alt.X("type").scale(zero=False),alt.Y("rating_numeric"))       
       
    st.altair_chart(base, use_container_width=True)



pg = st.navigation({
   "Netflix Dataset - Group 3":[
      st.Page(page_introduction, title="Introduction"),
      st.Page(page1, title="Data Table"),
      st.Page(page2, title="Described data"),
    ],
    "Chart 1: Distribution of TV Shows and Movies by Type":[
       st.Page(page_Chart_1, title='Chart 1: Distribution of TV Shows and Movies by Type')
    ],
    "Chart 2: Relationship between Release Year and Duration":[
       st.Page(page_Chart_2_1, title="Chart 2 bar chart for duration per genre"),
       st.Page(page_Chart_2_2, title="Chart 2: Regression charts")
    ],
    "Chart 3: Distribution of Content by Country":[
       st.Page(page_Chart_3, title="Chart 3: Distribution of Content by Country") 
    ],
    "Chart 4: Distribution of Content by Rating": [
       st.Page(page_Chart_4_1, title="Chart 4: Distribution of Content by Rating"),
       st.Page(page_chart_4_2, title="Chart 4: Movies and TV Shows Clusters")
    ],
    })

pg.run()
st.image('./images/netflix banner.png')