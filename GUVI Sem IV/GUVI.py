import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

sns.set(style="whitegrid")

os.makedirs("charts", exist_ok=True)

netflix_df = pd.read_csv("netflix_titles.csv")
spotify_df = pd.read_csv("data.csv")

netflix_df.dropna(subset=['title', 'type', 'release_year'], inplace=True)
netflix_df.drop_duplicates(inplace=True)
netflix_df['release_year'] = netflix_df['release_year'].astype(int)
netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')

netflix_df['country'] = netflix_df['country'].fillna("Unknown").str.title()
netflix_df['country'] = netflix_df['country'].str.split(',').str[0].str.strip()

netflix_df['type'] = netflix_df['type'].str.strip().str.title()
netflix_df['duration'] = netflix_df['duration'].fillna('0 Unknown')

type_counts = netflix_df['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
fig = px.bar(type_counts, x='type', y='count', title="Netflix Content Type Distribution", color='type')

netflix_df['content_age'] = pd.Timestamp.now().year - netflix_df['release_year']
netflix_df['added_year'] = netflix_df['date_added'].dt.year
netflix_df['duration_int'] = pd.to_numeric(netflix_df['duration'].str.extract(r'(\d+)', expand=False), errors='coerce')
netflix_df['duration_type'] = netflix_df['duration'].str.extract(r'([a-zA-Z]+)', expand=False).str.title()
netflix_df['duration_type'] = netflix_df['duration_type'].replace({'Min': 'Minutes', 'Season': 'Seasons'})

spotify_df.dropna(inplace=True)
spotify_df.drop_duplicates(inplace=True)
spotify_df['duration_min'] = spotify_df['duration_ms'] / 60000
spotify_df = spotify_df[spotify_df['duration_min'] < 15]

q1 = spotify_df['popularity'].quantile(0.25)
q3 = spotify_df['popularity'].quantile(0.75)
iqr = q3 - q1
spotify_df = spotify_df[(spotify_df['popularity'] >= q1 - 1.5 * iqr) & (spotify_df['popularity'] <= q3 + 1.5 * iqr)]

q1_duration = spotify_df['duration_min'].quantile(0.25)
q3_duration = spotify_df['duration_min'].quantile(0.75)
iqr_duration = q3_duration - q1_duration
spotify_df = spotify_df[(spotify_df['duration_min'] >= q1_duration - 1.5 * iqr_duration) & (spotify_df['duration_min'] <= q3_duration + 1.5 * iqr_duration)]

q1_age = netflix_df['content_age'].quantile(0.25)
q3_age = netflix_df['content_age'].quantile(0.75)
iqr_age = q3_age - q1_age
netflix_df = netflix_df[(netflix_df['content_age'] >= q1_age - 1.5 * iqr_age) & (netflix_df['content_age'] <= q3_age + 1.5 * iqr_age)]

spotify_df['popularity_band'] = pd.cut(spotify_df['popularity'], bins=[0, 40, 70, 100], labels=['Low', 'Medium', 'High'])
spotify_df['energy_band'] = pd.cut(spotify_df['energy'], bins=[0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High'])

print("\nNetflix Type Distribution:\n", netflix_df['type'].value_counts())
print("\nTop 5 Netflix Countries:\n", netflix_df['country'].value_counts().head())
print("\nNetflix Content Age Stats:\n", netflix_df['content_age'].describe())
print("\nSpotify Track Popularity Stats:\n", spotify_df['popularity'].describe())
print("\nSpotify Track Duration (min):\n", spotify_df['duration_min'].describe())

print("\n🎯Insight: Most Netflix content is Movies, with USA leading in content production.")
print("📊Insight: Majority of Spotify tracks are between 2-5 mins and fall under medium popularity.")
print("🗑️Outliers removed from Spotify data using IQR method for 'popularity' and 'duration_min'.")
print("📈 Netflix content is steadily growing till recent years; older content (~10+ years) is less frequent.")


netflix_df.to_csv("cleaned_netflix.csv", index=False)
spotify_df.to_csv("cleaned_spotify.csv", index=False)

fig = px.bar(type_counts, x='type', y='count', labels={'type': 'Content Type', 'count': 'Count'}, title="Netflix Content Type Distribution", color='type', color_continuous_scale='Viridis')
fig.write_html("charts/netflix_type_dist_interactive.html")

top_countries = netflix_df['country'].value_counts().head(10).reset_index()
top_countries.columns = ['country', 'count']
fig = px.bar(top_countries, x='country', y='count', labels={'country': 'Country', 'count': 'Number of Titles'}, title="Top 10 Countries with Netflix Content", color='count', color_continuous_scale='Blues')
fig.write_html("charts/netflix_top_countries_interactive.html")

fig = px.histogram(netflix_df, x="release_year", nbins=30, title="Netflix Titles by Release Year", labels={"release_year": "Release Year"}, color_discrete_sequence=["salmon"])
fig.write_html("charts/netflix_release_year_interactive.html")

fig = px.histogram(netflix_df, x="content_age", nbins=30, title="Distribution of Netflix Content Age", labels={"content_age": "Content Age (Years)"}, color_discrete_sequence=["lightblue"])
fig.write_html("charts/netflix_content_age_interactive.html")

fig = px.scatter(spotify_df, x='energy', y='popularity', color='popularity_band', title="Spotify: Energy vs Popularity", labels={'energy': 'Energy', 'popularity': 'Popularity'})
fig.write_html("charts/spotify_energy_popularity_interactive.html")

fig = px.box(spotify_df, y='popularity', title="Boxplot of Spotify Popularity")
fig.write_html("charts/spotify_popularity_boxplot_interactive.html")

if 'genre' in spotify_df.columns:
    top_genres = spotify_df['genre'].value_counts().head(10)
    fig = px.pie(top_genres, names=top_genres.index, values=top_genres.values, title="Top 10 Spotify Genres")
    fig.write_html("charts/spotify_top_genres_pie_interactive.html")

features = ['popularity', 'energy', 'duration_min', 'danceability', 'valence']
fig = px.imshow(spotify_df[features].corr(), title="Spotify Track Feature Correlation", color_continuous_scale='YlGnBu')
fig.write_html("charts/spotify_correlation_heatmap_interactive.html")

with open("charts/summary.txt", "w") as f:
    f.write("Netflix Most Common Type: " + netflix_df['type'].mode()[0] + "\n")
    f.write("Top Country: " + netflix_df['country'].value_counts().idxmax() + "\n")
    f.write("Spotify Popularity Mean: " + str(round(spotify_df['popularity'].mean(), 2)) + "\n")
    f.write("Spotify Energy Mean: " + str(round(spotify_df['energy'].mean(), 2)) + "\n")
    f.write("Cleaned datasets saved as cleaned_netflix.csv and cleaned_spotify.csv\n")



print("✅ All charts are saved interactively and datasets are cleaned.")
