import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_powerbi(df):
    """Export data for Power BI"""
    Path("outputs").mkdir(exist_ok=True)
    
    column_order = [
        "show_id", "type", "title", "director", "cast", "country", 
        "country_first", "release_year", "rating", "duration", 
        "duration_value", "listed_in", "description", "text", 
        "year_added", "month_added", "has_director", "has_cast", "genre_count"
    ]
    
    cols = [c for c in column_order if c in df.columns]
    
    output_file = "outputs/cleaned_netflix_powerbi.csv"
    df[cols].to_csv(output_file, index=False, encoding="utf-8-sig")
    
    logger.info(f"💾 Export PowerBI: {output_file} ({len(df):,} rows)")

def export_summary_stats(df):
    """Export summary statistics"""
    stats = {
        'Metric': [
            'Total Content',
            'Movies',
            'TV Shows',
            'Countries',
            'Unique Titles',
            'Earliest Release',
            'Latest Release',
            'Avg Movie Duration (min)',
            'Most Common Rating',
            'With Director Info',
            'With Cast Info',
            'Avg Genres per Title'
        ],
        'Value': [
            len(df),
            len(df[df['type'] == 'Movie']),
            len(df[df['type'] == 'TV Show']),
            df['country_first'].nunique(),
            df['title'].nunique(),
            int(df['release_year'].min()),
            int(df['release_year'].max()),
            f"{df[df['type'] == 'Movie']['duration_value'].mean():.1f}",
            df['rating'].mode()[0] if len(df['rating'].mode()) > 0 else 'N/A',
            f"{(df['has_director'].sum() / len(df) * 100):.1f}%",
            f"{(df['has_cast'].sum() / len(df) * 100):.1f}%",
            f"{df['genre_count'].mean():.1f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    output_file = "outputs/summary_statistics.csv"
    stats_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"📊 Export statistics: {output_file}")
    
    return stats_df