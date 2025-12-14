import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm
from src.inference import load_model, get_recommendations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_all_recommendations():
    """Export all recommendations"""
    logger.info("\n" + "="*60)
    logger.info("📤 Export All Recommendations")
    logger.info("="*60 + "\n")
    
    df, sim, index_map = load_model()
    
    all_recs = []
    titles = list(index_map.keys())
    
    logger.info(f"🚀 Generating recommendations for {len(titles):,} titles...\n")
    
    for title in tqdm(titles, desc="Processing"):
        recs = get_recommendations(title, df, sim, index_map, top_k=5)
        if recs is not None:
            recs = recs.copy()
            recs['source_title'] = title
            all_recs.append(recs[['source_title', 'title', 'type', 
                                   'similarity_score', 'listed_in']])
    
    if all_recs:
        final = pd.concat(all_recs, ignore_index=True)
        output_file = "outputs/netflix_recommendations.csv"
        final.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"\n✅ Export successful!")
        logger.info(f"   📁 {output_file}")
        logger.info(f"   📊 {len(final):,} rows")
        logger.info(f"   💾 {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("="*60 + "\n")
    else:
        logger.error("❌ Could not generate recommendations\n")

def export_sample(n=100):
    """Export sample n recommendations"""
    logger.info(f"\n📤 Export Sample {n} Recommendations\n")
    
    df, sim, index_map = load_model()
    sample_titles = df.sample(n)['title'].tolist()
    
    all_recs = []
    for title in tqdm(sample_titles, desc="Processing"):
        recs = get_recommendations(title, df, sim, index_map, top_k=5)
        if recs is not None:
            recs = recs.copy()
            recs['source_title'] = title
            all_recs.append(recs)
    
    if all_recs:
        final = pd.concat(all_recs, ignore_index=True)
        output_file = f"outputs/sample_{n}_recommendations.csv"
        final.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n✅ Export successful: {output_file}\n")

if __name__ == "__main__":
    print("\nSelect:")
    print("  1. Export all (takes time)")
    print("  2. Export sample 100 titles")
    print("  3. Export sample 500 titles")
    
    choice = input("\nSelect (1-3): ").strip()
    
    if choice == '1':
        confirm = input("⚠️  This will take time. Confirm? (y/n): ").strip().lower()
        if confirm == 'y':
            export_all_recommendations()
    elif choice == '2':
        export_sample(100)
    elif choice == '3':
        export_sample(500)
    else:
        print("❌ Please select 1-3")