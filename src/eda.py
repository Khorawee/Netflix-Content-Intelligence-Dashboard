import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def save_plot(filename):
    """Save plot to file with error handling."""
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved plot: {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save plot {filename}: {e}")
        plt.close()

def plot_top_genres(df, out="outputs/plots/top_genres.png"):
    """Plot top 20 genres."""
    try:
        genres = df["listed_in"].dropna().str.split(",").explode().str.strip()
        top = genres.value_counts().head(20).reset_index()
        top.columns = ['Genre', 'Count']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x='Count', y='Genre', hue='Genre', legend=False, palette='viridis')
        plt.title("Top 20 Netflix Genres", fontsize=15, fontweight='bold')
        plt.xlabel("Number of Titles")
        plt.ylabel("")
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot genres: {e}")

def plot_top_countries(df, out="outputs/plots/top_countries.png"):
    """Plot top 20 producing countries."""
    try:
        top = df["country_first"].value_counts().head(20).reset_index()
        top.columns = ['Country', 'Count']
        top = top[top['Country'] != 'Unknown']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x='Count', y='Country', hue='Country', legend=False, palette='magma')
        plt.title("Top Producing Countries", fontsize=15, fontweight='bold')
        plt.xlabel("Number of Titles")
        plt.ylabel("")
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot countries: {e}")

def plot_content_type(df, out="outputs/plots/content_type.png"):
    """Plot content type distribution."""
    try:
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        plt.figure(figsize=(8, 6))
        colors = ['#e50914', "#5e5e5e"]
        plt.pie(type_counts['Count'], labels=type_counts['Type'], autopct='%1.1f%%',
                startangle=90, colors=colors, textprops={'fontsize': 12})
        plt.title("Content Type Distribution", fontsize=15, fontweight='bold')
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot content type: {e}")

def plot_ratings_distribution(df, out="outputs/plots/ratings.png"):
    """Plot ratings distribution."""
    try:
        ratings = df['rating'].value_counts().head(15).reset_index()
        ratings.columns = ['Rating', 'Count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=ratings, x='Count', y='Rating', hue='Rating', 
                   legend=False, palette='rocket')
        plt.title("Top 15 Content Ratings", fontsize=15, fontweight='bold')
        plt.xlabel("Number of Titles")
        plt.ylabel("")
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot ratings: {e}")

def plot_release_year_trend(df, out="outputs/plots/release_trend.png"):
    """Plot release year trend."""
    try:
        year_counts = df['release_year'].value_counts().sort_index()
        year_counts = year_counts[year_counts.index >= 1990]
        
        plt.figure(figsize=(12, 6))
        plt.plot(year_counts.index, year_counts.values, linewidth=2, color='#e50914')
        plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#e50914')
        plt.title("Content Release Trend (1990+)", fontsize=15, fontweight='bold')
        plt.xlabel("Year")
        plt.ylabel("Number of Titles")
        plt.grid(True, alpha=0.3)
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot release trend: {e}")

def plot_duration_distribution(df, out="outputs/plots/duration.png"):
    """Plot movie duration distribution."""
    try:
        movies = df[df['type'] == 'Movie']['duration_value'].dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(movies, bins=30, color='#e50914', alpha=0.7, edgecolor='black')
        plt.axvline(movies.median(), color='yellow', linestyle='--', 
                   linewidth=2, label=f'Median: {movies.median():.0f} min')
        plt.title("Movie Duration Distribution", fontsize=15, fontweight='bold')
        plt.xlabel("Duration (minutes)")
        plt.ylabel("Frequency")
        plt.legend()
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot duration: {e}")

def plot_added_by_year(df, out="outputs/plots/added_trend.png"):
    """Plot content added to Netflix by year."""
    try:
        added_counts = df['year_added'].value_counts().sort_index()
        added_counts = added_counts[added_counts.index >= 2010]
        
        plt.figure(figsize=(12, 6))
        plt.bar(added_counts.index, added_counts.values, color='#221f1f', alpha=0.8)
        plt.title("Content Added to Netflix by Year", fontsize=15, fontweight='bold')
        plt.xlabel("Year Added")
        plt.ylabel("Number of Titles")
        plt.xticks(rotation=45)
        save_plot(out)
    except Exception as e:
        logger.error(f"❌ Failed to plot added trend: {e}")

def generate_all_plots(df):
    """Generate all standard plots."""
    logger.info("\n📊 Generating plots...")
    
    plots = [
        ("Content Type", plot_content_type),
        ("Top Genres", plot_top_genres),
        ("Top Countries", plot_top_countries),
        ("Ratings", plot_ratings_distribution),
        ("Release Trend", plot_release_year_trend),
        ("Duration", plot_duration_distribution),
        ("Added Trend", plot_added_by_year)
    ]
    
    for name, plot_func in plots:
        try:
            plot_func(df)
        except Exception as e:
            logger.error(f"❌ Failed to generate {name}: {e}")
    
    logger.info("✅ All plots generated successfully.\n")