import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def validate_data(df: pd.DataFrame) -> dict:
    """Validate data quality and return report."""
    report = {
        'total_rows': len(df),
        'duplicates': df.duplicated(subset=['show_id']).sum(),
        'missing_critical': {},
        'invalid_years': 0,
        'empty_text': 0
    }
    
    critical_cols = ['show_id', 'title', 'type']
    for col in critical_cols:
        if col in df.columns:
            report['missing_critical'][col] = df[col].isnull().sum()
    
    if 'release_year' in df.columns:
        invalid_years = df['release_year'].apply(
            lambda x: x < 1900 or x > 2030 if pd.notna(x) else False
        )
        report['invalid_years'] = invalid_years.sum()
    
    return report

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataframe for modeling and BI."""
    logger.info("🔄 Preprocessing data...")
    
    validation = validate_data(df)
    if validation['duplicates'] > 0:
        logger.warning(f"⚠️ Found {validation['duplicates']} duplicates")
    
    initial_count = len(df)
    df = df.drop_duplicates(subset=['show_id'], keep='first').copy()
    if len(df) < initial_count:
        logger.info(f"🗑️ Dropped {initial_count - len(df)} duplicate rows")
    
    df["country_first"] = df["country"].fillna("Unknown").apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Unknown"
    )
    
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    if df['year_added'].notna().any():
        median_year = df['year_added'].median()
        df['year_added'].fillna(median_year, inplace=True)
        df['month_added'].fillna(6, inplace=True)
    
    df["text"] = (
        df["title"].fillna("") + " " +
        df["listed_in"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["cast"].fillna("") + " " +
        df["director"].fillna("")
    ).apply(clean_text)
    
    empty_text = (df["text"].str.len() < 10).sum()
    if empty_text > 0:
        logger.warning(f"⚠️ Found {empty_text} items with minimal text content")
    
    df["rating"] = df["rating"].fillna("UR")
    df["duration"] = df["duration"].fillna("Unknown")
    df['duration_value'] = df['duration'].str.extract(r'(\d+)').astype(float)
    
    df.loc[df['release_year'] < 1900, 'release_year'] = df['release_year'].median()
    df.loc[df['release_year'] > 2030, 'release_year'] = df['release_year'].median()
    
    df['has_director'] = df['director'].notna().astype(int)
    df['has_cast'] = df['cast'].notna().astype(int)
    df['genre_count'] = df['listed_in'].fillna("").str.count(',') + 1
    
    before_clean = len(df)
    df = df.dropna(subset=['title', 'type'])
    if len(df) < before_clean:
        logger.info(f"🗑️ Removed {before_clean - len(df)} rows with missing critical data")
    
    logger.info(f"✅ Preprocessing complete: {len(df):,} rows remain")
    return df

def get_preprocessing_summary(df_before, df_after):
    """Display preprocessing summary."""
    logger.info("\n" + "="*60)
    logger.info("📋 Preprocessing Summary")
    logger.info("="*60)
    logger.info(f"Original Rows: {len(df_before):,}")
    logger.info(f"Processed Rows: {len(df_after):,}")
    logger.info(f"Rows Removed: {len(df_before) - len(df_after):,}")
    logger.info(f"Original Columns: {len(df_before.columns)}")
    logger.info(f"Final Columns: {len(df_after.columns)}")
    logger.info(f"New Columns Added: {len(df_after.columns) - len(df_before.columns)}")
    
    missing_after = df_after.isnull().sum()
    critical_missing = missing_after[missing_after > 0]
    if len(critical_missing) > 0:
        logger.info("\n⚠️ Remaining Missing Values:")
        for col, count in critical_missing.items():
            pct = (count / len(df_after)) * 100
            logger.info(f"   {col}: {count:,} ({pct:.1f}%)")
    
    logger.info("="*60 + "\n")