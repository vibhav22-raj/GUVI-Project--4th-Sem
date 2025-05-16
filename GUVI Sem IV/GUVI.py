import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot theme for visuals
sns.set(style="whitegrid", palette="muted")

# 1. Load Datasets
netflix_df = pd.read_csv("netflix_titles.csv")
spotify_df = pd.read_csv("data.csv")  # Update this if file name differs

# 2. Data Cleaning - Netflix
netflix_df.dropna(subset=['title', 'type', 'release_year'], inplace=True)
netflix_df.drop_duplicates(inplace=True)
netflix_df['release_year'] = netflix_df['release_year'].astype(int)
netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')
netflix_df['content_age'] = 2025 - netflix_df['release_year']
netflix_df['added_year'] = netflix_df['date_added'].dt.year
netflix_df['country'].fillna("Unknown", inplace=True)

netflix_df['duration_type'] = netflix_df['duration'].str.extract(r'(\D+)', expand=False)
netflix_df['duration_int'] = netflix_df['duration'].str.extract(r'(\d+)', expand=False).astype(float)

# 3. Data Cleaning - Spotify
spotify_df.dropna(inplace=True)
spotify_df.drop_duplicates(inplace=True)

spotify_df['duration_min'] = spotify_df['duration_ms'] / 60000

spotify_df = spotify_df[spotify_df['duration_min'] < 15]

q1 = spotify_df['popularity'].quantile(0.25)
q3 = spotify_df['popularity'].quantile(0.75)
iqr = q3 - q1
spotify_df = spotify_df[
    (spotify_df['popularity'] >= q1 - 1.5 * iqr) &
    (spotify_df['popularity'] <= q3 + 1.5 * iqr)
]

spotify_df['popularity_band'] = pd.cut(
    spotify_df['popularity'], bins=[0, 40, 70, 100], labels=['Low', 'Medium', 'High']
)
spotify_df['energy_band'] = pd.cut(
    spotify_df['energy'], bins=[0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High']
)

# 4. Save Cleaned Data for Power BI
netflix_df.to_csv("cleaned_netflix.csv", index=False)
spotify_df.to_csv("cleaned_spotify.csv", index=False)

# 5. Additional EDA Plots (export for Power BI design guidance)

# Netflix: Type count
plt.figure(figsize=(8, 4))
sns.countplot(data=netflix_df, x='type')
plt.title("Netflix Content Type Distribution")
plt.savefig("netflix_type_dist.png")
plt.close()

# Netflix: Top 10 Countries
top_countries = netflix_df['country'].value_counts().head(10)
top_countries.plot(kind='bar', title='Top 10 Countries with Netflix Content', color='skyblue')
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("netflix_top_countries.png")
plt.close()

# Netflix: Release trend
plt.figure(figsize=(10, 5))
sns.histplot(netflix_df['release_year'], bins=30, kde=True)
plt.title("Netflix Titles by Release Year")
plt.xlabel("Release Year")
plt.savefig("netflix_release_year.png")
plt.close()

# Spotify: Energy vs Popularity with popularity bands
plt.figure(figsize=(8, 4))
sns.scatterplot(data=spotify_df, x='energy', y='popularity', hue='popularity_band')
plt.title("Spotify: Energy vs Popularity")
plt.savefig("spotify_energy_popularity.png")
plt.close()

# Spotify: Distribution of genres (if available)
if 'genre' in spotify_df.columns:
    top_genres = spotify_df['genre'].value_counts().head(10)
    top_genres.plot(kind='bar', title='Top 10 Spotify Genres', color='orange')
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    plt.savefig("spotify_top_genres.png")
    plt.close()

print("✅ Enhanced data exported. Visuals generated for Power BI use.")
