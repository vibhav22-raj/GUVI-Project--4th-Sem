# ❇️ GUVI Project – 4th Semester  
## 🎬📽️🇳  Netflix & Spotify Data Analytics Project

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Power BI](https://img.shields.io/badge/Tool-Power%20BI-yellow)
![Libraries](https://img.shields.io/badge/Libraries-pandas%2C%20matplotlib%2C%20seaborn-lightgrey)
![Pyplot](https://img.shields.io/badge/Tool-Matplotlib%20Pyplot-darkgreen)

---

This project explores **data collection**, **preprocessing**, **exploratory data analysis (EDA)**, and **visualization** using **Python** and **Power BI**. It delivers actionable insights from **Netflix** and **Spotify** datasets through cleaning, feature engineering, outlier detection, and visual storytelling.

## 📦 Datasets Used

- `netflix_titles.csv` - Contains information about Netflix shows and movies.
- `data.csv` (Spotify) - Contains attributes like energy, popularity, and duration of Spotify tracks.


 # 🛠️ Project Setup
 ## 📁 Folder Structure
- *project/*
- │
- ├── netfix_titles.csv # Netflix data
- ├── data.csv # Spotify data
- ├── visuals/ # Saved charts and plots
- │ ├── netflix_type_dist.png
- │ ├── netflix_release_year.png
- │ ├── netflix_top_conutries.png
- │ └── spotify_energy_population.png
- ├── main.py # Main script for analysis
- └── README.md # This file

  ## ⚙️Requirements
  
  Install **Python 3.7** or above make sure the following libaries are installed:
  
  - *pip install pandas matplotlib seaborn*
  - *pip install plotly*


 ## ⚡How to Run the Project

1. Place the original datasets in the project folder:  
   - netflix_titles.csv  
   - data.csv

2. Run the Python Script:  
   - python main.py

3. The Script will:  
   - Clean and preprocess the datasets  
   - Create new features for analysis  
   - Handle missing values and outliers  
   - Generate and export cleaned CSVs for Power BI  
   - Save EDA visualizations to .png files


## 📊 Visualizations & Insights

| 🔍 Visualization                          | 📘 Insight |
|------------------------------------------|------------|
| **Netflix Type Distribution**            | Shows are more dominant than movies or vice versa |
| **Top 10 Netflix Countries**             | Identifies countries with highest Netflix content |
| **Release Year Histogram**               | Trends of content over years |
| **Spotify Energy vs Popularity**         | High-energy songs tend to be more popular |
| **Spotify Popularity Boxplot**           | Outliers and spread of popularity |
 

## 📊 Visualization Objectives

### 🔴 Netflix:
1. **Content Type Distribution**  
   → Understand proportion of Movies vs TV Shows

2. **Top 10 Content-Producing Countries**  
   → Identify which countries contribute the most

3. **Titles by Release Year**  
   → Explore production trends over the years

4. **Content Age Distribution**  
   → Analyze how old the available content is

### 🟢 Spotify:
1. **Energy vs Popularity Scatter Plot**  
   → Explore how song energy relates to popularity

2. **Popularity Boxplot**  
   → Visualize spread and outliers in popularity

3. **Top 10 Genres (if available)**  
   → Discover most frequent music genres

4. **Correlation Heatmap**  
   → Understand relationships among features like popularity, energy, duration, etc.

---

## 🖱️ How to Interact with the Charts

All charts are saved in the `charts/` folder as interactive `.html` files.

- **Open in Browser**  
  Simply double-click or right-click any `.html` file and open in your browser.

- **Hover Tooltips**  
  Hover over chart elements to see detailed information.

- **Zoom & Pan**  
  Use mouse scroll or drag to zoom and explore sections of the chart.

- **Legend Toggle**  
  Click on legend items (like "Movies") to show/hide data categories.

---

### 📊 Interactive Visualizations
- `charts/netflix_type_dist_interactive.html`
- `charts/netflix_top_countries_interactive.html`
- `charts/netflix_release_year_interactive.html`
- `charts/netflix_content_age_interactive.html`
- `charts/spotify_energy_popularity_interactive.html`
- `charts/spotify_popularity_boxplot_interactive.html`
- `charts/spotify_correlation_heatmap_interactive.html`



## 📈Output
   - -->cleaned_netflix.csv and cleaned_spotify.csv :Processed datasets ready for import into Power BI.
   - -->.png files:EDA plots like type distribution,top countries,energy vs popularity,etc.
   - -->html image that show different charts visualizing the data of the netflix and sportify

## 📊Power BI
  **Use Power BI to import the cleaned CSV files:**
  - Build interactive dashboards using the visual insights.
  - Combine filters, charts and slicers to explore trends in music and streaming.

## 🧠 Data Storytelling

We combined visualizations with statistical summaries to highlight key patterns:
- Rise of Netflix content post-2015.
- Streaming music trends based on song energy.
- Country-wise content distribution indicating regional preferences.
- Classification of songs into popularity bands.

## 📝Summary Report
- `charts/summary.txt` — Contains key statistics and insights from the cleaned data


# 💡Key Insights
 - -->Netflix mostly features content from the US and recent years.
 - -->Spotify tracks with high energy tend to be more popular.
 - -->Visulaizatins show trends in content release,genre distribution and user preferences.
 - -->HTML png that shows the different charts.

## 🎯 Project Goals Covered

- 🧹 *Data Cleaning*  
- 🛠 *Feature Engineering*  
- 🧩 *Handling Missing Values & Outliers*  
- 📊 *Visual Insights*  
- 📈 *Power BI Ready*
- 🧠 *Data Storytelling*

  
