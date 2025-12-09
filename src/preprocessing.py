"""
Improved data preprocessing for Netflix dataset
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetflixPreprocessor:
    """Enhanced preprocessing pipeline for Netflix data"""
    
    def __init__(self):
        # Columns ที่มีใน Netflix dataset จริง
        self.text_columns = [
            "title", "director", "cast", "country", 
            "date_added", "rating", "listed_in", "description"
        ]
        # คอลัมน์ที่เป็นตัวเลข
        self.numeric_columns = ["release_year"]
        # คอลัมน์ที่มี ID
        self.id_columns = ["show_id"]
        
    def clean_text(self, text: str, lowercase: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text)
        
        if lowercase:
            text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def parse_duration(self, duration_str: str) -> Dict[str, float]:
        """
        Parse duration string into minutes or seasons
        
        Args:
            duration_str: Duration like "90 min" or "2 Seasons"
            
        Returns:
            Dict with duration_min and duration_season
        """
        result = {"duration_min": np.nan, "duration_season": np.nan}
        
        if pd.isna(duration_str) or duration_str == "":
            return result
        
        duration_str = str(duration_str).strip()
        numbers = re.findall(r"(\d+)", duration_str)
        
        if not numbers:
            return result
        
        value = int(numbers[0])
        
        if "min" in duration_str.lower():
            result["duration_min"] = value
        elif "season" in duration_str.lower():
            result["duration_season"] = value
        
        return result
    
    def parse_date_added(self, date_str: str) -> Dict[str, float]:
        """
        Parse date_added into year and month
        
        Args:
            date_str: Date string
            
        Returns:
            Dict with added_year and added_month
        """
        try:
            dt = pd.to_datetime(date_str)
            return {
                "added_year": dt.year,
                "added_month": dt.month
            }
        except:
            return {
                "added_year": np.nan,
                "added_month": np.nan
            }
    
    def extract_keywords(self, text: str, max_words: int = 100) -> str:
        """
        Extract important keywords from text
        
        Args:
            text: Input text
            max_words: Maximum number of words to keep
            
        Returns:
            Cleaned text with important words
        """
        if not text:
            return ""
        
        # Remove common stopwords manually (basic set)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'his', 'her', 'its', 'their', 'as', 'by', 'from'
        }
        
        words = text.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Keep only max_words
        return " ".join(filtered_words[:max_words])
    
    def combine_features(self, row: pd.Series, 
                        columns: List[str] = None) -> str:
        """
        Combine multiple text columns into single feature string
        
        Args:
            row: DataFrame row
            columns: List of column names to combine
            
        Returns:
            Combined text string
        """
        if columns is None:
            columns = [
                "description_clean",
                "listed_in_clean", 
                "cast_clean",
                "director_clean",
                "country_clean"  # เพิ่ม country
            ]
        
        # Weight different features (description gets more weight)
        weights = {
            "description_clean": 3,  # Most important
            "listed_in_clean": 2,    # Genres are important
            "cast_clean": 1,
            "director_clean": 1,
            "country_clean": 1       # Country matters
        }
        
        combined_parts = []
        
        for col in columns:
            if col in row and pd.notna(row[col]) and row[col]:
                text = str(row[col])
                # Repeat based on weight to give more importance
                weight = weights.get(col, 1)
                combined_parts.extend([text] * weight)
        
        combined = " ".join(combined_parts)
        return self.extract_keywords(combined, max_words=200)
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline
        
        Args:
            df: Raw Netflix DataFrame
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting preprocessing for {len(df)} records...")
        df = df.copy()
        
        # 1. Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info("✓ Normalized column names")
        
        # 2. Fill NA values for text columns
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna("")
        logger.info("✓ Filled NA values")
        
        # 3. Clean text columns
        for col in ["title", "director", "cast", "country", "listed_in", "description"]:
            if col in df.columns:
                df[f"{col}_clean"] = df[col].apply(self.clean_text)
        logger.info("✓ Cleaned text columns")
        
        # 4. Parse duration
        if "duration" in df.columns:
            duration_df = df["duration"].apply(self.parse_duration).apply(pd.Series)
            df = pd.concat([df, duration_df], axis=1)
            logger.info("✓ Parsed duration")
        
        # 5. Parse date_added
        if "date_added" in df.columns:
            date_df = df["date_added"].apply(self.parse_date_added).apply(pd.Series)
            df = pd.concat([df, date_df], axis=1)
            logger.info("✓ Parsed dates")
        
        # 6. Ensure release_year is numeric
        if "release_year" in df.columns:
            df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
            logger.info("✓ Converted release_year to numeric")
        
        # 7. Create combined text features
        df["text_features"] = df.apply(self.combine_features, axis=1)
        logger.info("✓ Created combined text features")
        
        # 8. Add derived features
        if "description" in df.columns:
            df["description_length"] = df["description"].fillna("").str.len()
        
        if "cast" in df.columns:
            df["cast_count"] = df["cast"].fillna("").apply(
                lambda x: len(x.split(",")) if x else 0
            )
        
        if "director" in df.columns:
            df["has_director"] = df["director"].fillna("").apply(
                lambda x: 1 if x else 0
            )
        
        # 9. Remove duplicates based on title + type
        original_len = len(df)
        if "type" in df.columns:
            df = df.drop_duplicates(subset=["title", "type"], keep="first")
        else:
            df = df.drop_duplicates(subset=["title"], keep="first")
        if len(df) < original_len:
            logger.info(f"✓ Removed {original_len - len(df)} duplicate titles")
        
        # 10. Remove rows with empty text_features
        df = df[df["text_features"].str.strip() != ""]
        logger.info(f"✓ Final dataset: {len(df)} records")
        
        return df


# Convenience function for backward compatibility
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Netflix DataFrame
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Processed DataFrame
    """
    preprocessor = NetflixPreprocessor()
    return preprocessor.process_dataframe(df)


def clean_text(s: str) -> str:
    """Clean text string (backward compatibility)"""
    preprocessor = NetflixPreprocessor()
    return preprocessor.clean_text(s)