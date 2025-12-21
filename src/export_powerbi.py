import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_for_powerbi(text):
    """Clean text to prevent CSV parsing issues in Power BI"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    # แทนที่ newlines ด้วย space
    text = text.replace('\n', ' ').replace('\r', ' ')
    # ลบ tabs
    text = text.replace('\t', ' ')
    # ลบ multiple spaces
    text = ' '.join(text.split())
    # เอา quotes ออกเพื่อป้องกัน escape issues
    text = text.replace('"', "'")
    
    return text.strip()

def export_powerbi(df):
    """Export data for Power BI with robust CSV handling"""
    Path("outputs").mkdir(exist_ok=True)
    
    # สร้าง clean copy
    df_export = df.copy()
    
    # Clean text columns ที่มักมีปัญหา
    text_columns = ['description', 'cast', 'director', 'title', 'listed_in', 'text']
    for col in text_columns:
        if col in df_export.columns:
            logger.info(f"  Cleaning column: {col}")
            df_export[col] = df_export[col].apply(clean_for_powerbi)
    
    # เรียงลำดับ columns
    column_order = [
        "show_id", "type", "title", "director", "cast", "country", 
        "country_first", "release_year", "rating", "duration", 
        "duration_value", "listed_in", "description", "text", 
        "year_added", "month_added", "has_director", "has_cast", "genre_count"
    ]
    
    cols = [c for c in column_order if c in df_export.columns]
    
    output_file = "outputs/cleaned_netflix_powerbi.csv"
    
    # Export with explicit quoting and escaping
    df_export[cols].to_csv(
        output_file, 
        index=False, 
        encoding="utf-8-sig",
        quoting=1,  # QUOTE_ALL - quote all fields
        escapechar='\\',  # explicit escape character
        doublequote=True,  # double quotes for quotes
        lineterminator='\n'  # explicit line terminator
    )
    
    logger.info(f"💾 Export PowerBI: {output_file} ({len(df_export):,} rows)")
    
    # Validation check
    try:
        test_df = pd.read_csv(output_file, encoding='utf-8-sig')
        if len(test_df) == len(df_export):
            logger.info(f"✅ Validation passed: {len(test_df):,} rows")
        else:
            logger.warning(f"⚠️ Row mismatch: Expected {len(df_export):,}, got {len(test_df):,}")
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")

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