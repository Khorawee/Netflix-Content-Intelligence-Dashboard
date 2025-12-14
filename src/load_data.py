import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_netflix(path="data/netflix_titles.csv"):
    """Load Netflix dataset with encoding fallback handling."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Error: File not found at {file_path}. Please check the 'data' folder.")
    
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ Successfully loaded data using '{encoding}' encoding.")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            raise
            
    raise ValueError(f"❌ Failed to decode file {path} with common encodings.")

def get_data_info(df):
    """Display basic data information."""
    logger.info("\n" + "="*60)
    logger.info("📊 Basic Information")
    logger.info("="*60)
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"\nColumn Names: {list(df.columns)}")
    logger.info(f"\nData Types:\n{df.dtypes}")
    logger.info(f"\nMissing Values:\n{df.isnull().sum()}")
    logger.info(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    logger.info("="*60 + "\n")