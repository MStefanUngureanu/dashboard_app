import streamlit as st
from collections import Counter
import pandas as pd
import numpy as np

st.title("Movie Dashboard")
st.write("An interactive dashboard for 18 to 35 movie lovers.")
st.write("This dashboard is designed to help you find the best movies to watch based on your preferences.")
# cache data
@st.cache_data
#importing data
def load_data():
    df_mov = pd.read_csv("movies.csv", sep=',', encoding='latin-1')
    df_rt = pd.read_csv("rating.csv", sep=',', encoding='latin-1')
    df_tag = pd.read_csv("tags.csv", sep=',', encoding='latin-1')
    df_mov['year'] = df_mov['title'].str.extract(r'\((\d{4})\)').astype('Int64')

    # Drop duplicates
    df_mov.drop_duplicates(inplace=True)
    df_rt.drop_duplicates(inplace=True)
    df_tag.drop_duplicates(inplace=True)

    # Drop rows with missing critical values
    df_mov.dropna(subset=['movieId', 'title'], inplace=True)
    df_rt.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)
    df_tag.dropna(subset=['userId', 'movieId', 'tag'], inplace=True)

    # Convert timestamps
    df_rt['timestamp'] = pd.to_datetime(df_rt['timestamp'], unit='ms')
    df_tag['timestamp'] = pd.to_datetime(df_tag['timestamp'], unit='ms')
    return df_mov, df_rt, df_tag

df_mov, df_rt, df_tag = load_data() # load data and cache it, seems important to use the cache decorator

st.divider()
# Split genres
def normalize_genres(val):
    if isinstance(val, list):
        return '|'.join(val)
    elif isinstance(val, str):
        return val
    else:
        return None
df_mov['genres'] = df_mov['genres'].apply(normalize_genres)
# https://cheat-sheet.streamlit.app/
# https://docs.streamlit.io/develop/api-reference

st.subheader("Some database statistics")
df_clean = df_mov.dropna(subset=['genres'])
df_clean = df_clean[df_clean['genres'] != '(no genres listed)']
all_genres = df_clean['genres'].str.split('|').explode().dropna()

# print("All genres " + str(all_genres) )
genre_counts = Counter(all_genres)
top_5_genres = genre_counts.most_common(5)
#genre_exploded = df_mov.explode('genres')

total_movies = df_mov['movieId'].nunique()
total_users = df_rt['userId'].nunique()
total_ratings = len(df_rt)
#top_genre = genre_exploded['genres'].value_counts().idxmax()
top_5_string = ', '.join([f"{genre} ({count})" for genre, count in top_5_genres])

#print("How many nan? " + str(df_mov['genres'].isna().sum()))        # how many NaNs?
#print(df_mov['genres'].head())              # sample of genres
#print(all_genres.head(3))                  # explode results

#print(df_mov['genres'].head(10))
#print(df_mov['genres'].apply(type).value_counts())

#print(df_mov['genres'].unique())            # unique genres 
st.markdown("### Top 5 Movie Genres by count:")
st.markdown(top_5_string)

average_rating = df_rt['rating'].mean()
st.metric("Average movie rating:", f"{average_rating:.2f}")
st.metric("Totale number of genres: ",len(genre_counts))
avg_movies_per_user = df_rt.groupby('userId')['movieId'].nunique().mean()
st.metric("Average movies rated per capita: ", f"{avg_movies_per_user:.1f}")
print(df_rt['userId'].nunique())
print(len(df_rt['movieId']))
# avg_movies_per_user = total_ratings / total_users

st.divider()
# showing how many movies users have rated in total.

movies_per_user = df_rt.groupby('userId')['movieId'].nunique()
movies_per_user_df = movies_per_user.reset_index()
movies_per_user_df.columns = ['userId', 'movies rated']
st.scatter_chart(movies_per_user_df, x='userId', y='movies rated', color =  "#2cb28c" )

st.divider()

# group ratings with movie year
st.subheader("Plot filtering by year")
df_merged = df_rt.merge(df_mov[['movieId', 'year']], on='movieId')
# add a slider
data_min_year = int(df_merged['year'].min())
data_max_year = int(df_merged['year'].max())

# Manual Min/Max Year
cminy = st.number_input(
    "Enter minimum year for range", min_value=data_min_year, max_value=data_max_year, value=data_min_year, step=1
)
cmaxy = st.number_input(
    "Enter maximum year for range", min_value=cminy, max_value=data_max_year, value=data_max_year, step=1
)

year_range = st.slider(
    "SelectYear Range",
    min_value=cminy,
    max_value=cmaxy,
    value=(cminy, cmaxy),
    step=1
)
# Filter by year range
df_filtered = df_merged[(df_merged['year'] >= year_range[0]) & (df_merged['year'] <= year_range[1])
]
# Group by user and count movies rated
movies_per_user = df_filtered.groupby('userId')['movieId'].nunique().reset_index()
movies_per_user.columns = ['userId', 'movies rated']
movies_per_user['user'] = range(1, len(movies_per_user) + 1) # will rename the users for smaller numbers

# Display scatter chart
st.scatter_chart(movies_per_user, x='user', y='movies rated')





print("Average movies rated / per User:", avg_movies_per_user)
st.subheader("Available movie statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies", total_movies)
col2.metric("Unique Users", total_users)
col3.metric("Total Ratings", total_ratings)
col4.metric("Top Genre", top_5_genres[0][0])

st.divider()
# Flatten list of lists and get unique genres
#all_genres = df_mov['genres'].apply(normalize_genres)
all_genres = set()
df_mov['genres'].dropna().str.split('|').apply(all_genres.update)

# Sort for display
sorted_genres = sorted(all_genres)

# Checkbox to show them
if st.checkbox("Show all available genres"):
    st.markdown("Available Genres:")
    st.write(sorted_genres)
st.divider()    
# add a genre  reccomandations searchbar


st.subheader("Movie Recommendations Search" )

df_combined = df_rt.merge(df_mov, on='movieId') # merge movie ids with ratings
genre_input = st.text_input("Enter a genre (e.g. Comedy, Action, Drama): ") #prompts the user for input, hopefully an available movie genre
sort_order = st.radio("Sort by rating:", options=["Highly rated ", "Poorly rated"]) #will sort the results by rating, either highly rated or poorly rated
# poorly rated movies to be avoided.
if genre_input:
    # Filter movies that contain the genre 
    genre_filtered = df_combined[df_combined['genres'].str.contains(genre_input, case=False, na=False)]

    # Calculate average rating per movie
    movie_ratings = genre_filtered.groupby(['movieId', 'title'])['rating'].mean().reset_index()
    movie_ratings.rename(columns={'rating': 'Average Rating'}, inplace=True)

    # Sort by rating
    ascending = True if sort_order == "Poorly rated" else False
    movie_ratings = movie_ratings.sort_values(by='Average Rating', ascending=ascending)

    # Show top N results
    top_n = st.slider("Number of recommendations", 5, 20, 10) #user can choose how many recommendations they want to see
    # Display results
    st.write(f"Top {top_n} movies in genre **{genre_input.capitalize()}**:")
    st.dataframe(movie_ratings.head(top_n))

    if movie_ratings.empty:
        st.warning("No movies found for that genre.")

# show user tags for entered movie, show what people are saying about the movie, aggregated

st.divider() 

# Merge tags with movie titles
df_tags_with_titles = df_tag.merge(df_mov[['movieId', 'title']], on='movieId', how='left')
# Merge genres into tag_summary using movieId
DFT = df_tags_with_titles.merge(df_mov[['movieId', 'genres']], on='movieId', how='left')



movies= df_mov['movieId'].tolist() # movies list for the search bar
# --- Initialize session state for search input ---
if 'search_input' not in st.session_state:
    st.session_state['search_input'] = ""

import random
def pick_random():
    st.session_state['search_input'] = random.choice(movies)

st.button("No ideeas? - 🎲 Pick a Random Movie or ", on_click=pick_random)
search_input = st.text_input("Copy a movie by title or ID to view user comments:", value=st.session_state['search_input'])
# Search input for title or ID
#search_input = st.text_input("Copy a movie by title or ID to view user comments:")
if search_input:
    def get_star_rating_html(rating, size=24): # this will show the movie rating visually as a group of hearts, full, empty or half/broken.
        full = "❤️"
        half = "💔"  # Broken heart as "half"
        empty = "🤍"  # White heart
        rounded = round(rating * 2) / 2
        full_stars = int(rounded)
        half_star = 1 if rounded - full_stars == 0.5 else 0
        empty_stars = 5 - full_stars - half_star

        # Stars HTML for quick visual rating , did not work
        #full = f"<img src='sfull.png' width='20' style='margin-right:2px;'/>"
        #half = f"<img src='shalf.png' width='20' style='margin-right:2px;'/>"
        #empty = f"<img src='sempty.png' width='20' style='margin-right:2px;'/>"
        return full * full_stars + half * half_star + empty * empty_stars
    
    # Check if input is userid or text 
    if search_input.isdigit():
        movie_id = int(search_input)
        filtered_tags = DFT[DFT['movieId'] == movie_id]
    else:
        # Case-insensitive title search
        filtered_tags = DFT[DFT['title'].str.contains(search_input, case=False, na=False)]
    
    if filtered_tags.empty:
        st.warning("No tags found for that movie or wrong name / ID. Please try again.")
    else:
        # Group tags by movie 
        tag_summary = (
            filtered_tags
            .groupby(['movieId', 'title'])['tag'] #groups all tag entries for each movie
            .apply(lambda tags: list(tags.unique())) # gets only the unique tags (removes duplicates)
            .reset_index()
        )
        
        # Show results
        for _, row in tag_summary.iterrows(): # loop through DataFrame rows
            movie_id = row['movieId']
            avg_rating = df_rt[df_rt['movieId'] == movie_id]['rating'].mean()
            stars_html = get_star_rating_html(avg_rating)
            title = row['title']
            tags = row['tag']
           
            # Get rating count
            rating_count = df_rt[df_rt['movieId'] == movie_id].shape[0] #count matching rows for the movie ID 
            # Get genres from DFT
            genres = DFT[DFT['movieId'] == movie_id]['genres'].values
            genres_text = genres[0].replace('|', ', ') if len(genres) > 0 else 'N/A'
            # Display movie information
            st.markdown(f"{title}  (ID: **{movie_id}**)  {stars_html}", unsafe_allow_html=True)
            st.markdown(f"**Genres:** {genres_text}")
            st.markdown(f"**Number of Ratings:** {rating_count}")
            st.markdown("**User Tags:**")
            st.write(", ".join(tags))
    
    # further improvements can track what users are looking for, movies or genres, movie popularity can be tracked 
    #over time and so on.