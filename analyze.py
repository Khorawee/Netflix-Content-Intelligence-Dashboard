import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from src.inference import load_model, get_recommendations, search_titles
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interactive_recommend():
    """Interactive Recommendation System"""
    logger.info("\n" + "="*70)
    logger.info("🎬 Netflix Recommendation System - Interactive Mode")
    logger.info("="*70 + "\n")
    
    df, sim, index_map = load_model()
    
    while True:
        print("\nOptions:")
        print("  1. Search titles")
        print("  2. Get recommendations")
        print("  3. Show sample titles")
        print("  4. Exit")
        
        choice = input("\nSelect (1-4): ").strip()
        
        if choice == '1':
            query = input("Search: ").strip()
            results = search_titles(query, df)
            print(f"\n🔍 Found {len(results)} results:")
            print(results.to_string(index=False))
            
        elif choice == '2':
            title = input("Title: ").strip()
            print(f"\n🎯 Recommendations for: {title}")
            print("-" * 70)
            
            recs = get_recommendations(title, df, sim, index_map, top_k=5)
            if recs is not None:
                for idx, row in recs.iterrows():
                    print(f"\n{idx+1}. {row['title']} ({row['release_year']})")
                    print(f"   Type: {row['type']} | Rating: {row['rating']}")
                    print(f"   Score: {row['similarity_score']:.4f}")
                    print(f"   Genre: {row['listed_in']}")
            else:
                print("\n❌ Title not found. Try searching first.")
                    
        elif choice == '3':
            sample = df.sample(20)['title'].tolist()
            print("\n📝 Sample Titles:")
            for i, title in enumerate(sample, 1):
                print(f"   {i}. {title}")
                
        elif choice == '4':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Please select 1-4")

def analyze_specific_title(title):
    """Analyze specific title"""
    df, sim, index_map = load_model()
    
    print(f"\n🔎 Analyzing: {title}")
    print("="*70)
    
    if title not in index_map:
        print("❌ Title not found")
        print("\n💡 Try searching instead:")
        results = search_titles(title, df)
        if len(results) > 0:
            print(results[['title', 'type', 'release_year']].head(10).to_string(index=False))
        return
    
    idx = index_map[title]
    content = df.iloc[idx]
    
    print(f"\n📌 Information:")
    print(f"   Title: {content['title']}")
    print(f"   Type: {content['type']}")
    print(f"   Release: {content['release_year']}")
    print(f"   Rating: {content['rating']}")
    print(f"   Genre: {content['listed_in']}")
    
    if 'description' in content and pd.notna(content['description']):
        desc = content['description']
        print(f"   Description: {desc[:150]}...")
    
    print(f"\n🎯 Top 10 Similar Recommendations:")
    print("-" * 70)
    
    recs = get_recommendations(title, df, sim, index_map, top_k=10)
    if recs is not None:
        print(recs[['title', 'type', 'release_year', 'similarity_score', 'listed_in']].to_string(index=False))
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
        analyze_specific_title(title)
    else:
        interactive_recommend()