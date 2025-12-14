import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from src.load_data import load_netflix, get_data_info
from src.preprocess import preprocess, get_preprocessing_summary
from src.eda import generate_all_plots
from src.export_powerbi import export_powerbi, export_summary_stats
from src.model_tfidf import build_tfidf, analyze_model_performance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*70)
    logger.info("🎬 Netflix Data Science Project - Recommendation System")
    logger.info("="*70 + "\n")
    
    try:
        # 1. Load Data
        logger.info("📥 Step 1: Loading Data")
        logger.info("-" * 70)
        df_raw = load_netflix()
        if df_raw is None:
            return
        get_data_info(df_raw)
        
        # 2. Preprocess
        logger.info("🔧 Step 2: Data Preprocessing")
        logger.info("-" * 70)
        df = preprocess(df_raw)
        get_preprocessing_summary(df_raw, df)
        
        # 3. EDA
        logger.info("📊 Step 3: Exploratory Data Analysis")
        logger.info("-" * 70)
        generate_all_plots(df)
        
        # 4. Export
        logger.info("💾 Step 4: Exporting Data")
        logger.info("-" * 70)
        export_powerbi(df)
        stats_df = export_summary_stats(df)
        logger.info("")
        
        # 5. Build Model
        logger.info("🤖 Step 5: Building Recommendation Model")
        logger.info("-" * 70)
        
        mlflow.set_experiment("Netflix_Recommendation")
        with mlflow.start_run():
            vectorizer, sim = build_tfidf(df)
            
            metrics = analyze_model_performance(df, sim)
            
            mlflow.log_param("total_items", len(df))
            mlflow.log_param("tfidf_max_features", 5000)
            mlflow.log_param("unique_titles", df['title'].nunique())
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        
        # Summary
        logger.info("="*70)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\n📂 Output Files:")
        logger.info("   ├── 📊 outputs/plots/ (7 plots)")
        logger.info("   ├── 💾 outputs/cleaned_netflix_powerbi.csv")
        logger.info("   ├── 📈 outputs/summary_statistics.csv")
        logger.info("   └── 🤖 outputs/models/ (3 files)")
        logger.info("\n💡 Next Steps:")
        logger.info("   • Run 'python analyze.py' to test model")
        logger.info("   • Run 'python export_recs.py' to export recommendations")
        logger.info("   • Open Power BI and import cleaned_netflix_powerbi.csv")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()