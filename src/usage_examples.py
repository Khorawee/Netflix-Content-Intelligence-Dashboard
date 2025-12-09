"""
Usage examples for Netflix Recommender System
Run these examples after building the models with main.py
"""

from src.load_data import load_netflix
from src.preprocessing import preprocess
from src.recommender_tf import USERecommender
import pandas as pd


def example_1_basic_recommendation():
    """Example 1: Basic recommendation by title"""
    print("\n" + "="*60)
    print("Example 1: Basic Recommendation")
    print("="*60)
    
    # Load data
    df = load_netflix()
    df = preprocess(df)
    
    # Initialize recommender
    recommender = USERecommender()
    recommender.load_embeddings()
    recommender.load_similarity_matrix()
    recommender.load_index_map()
    
    # Get recommendations
    title = "Stranger Things"
    recommendations = recommender.recommend(
        title, 
        df, 
        top_n=5,
        return_scores=True
    )
    
    print(f"\nRecommendations for '{title}':\n")
    for idx, row in recommendations.iterrows():
        print(f"{idx+1}. {row['title']} (Score: {row['similarity_score']:.3f})")
        print(f"   {row['description'][:100]}...")
        print()


def example_2_text_query():
    """Example 2: Recommendation by text description"""
    print("\n" + "="*60)
    print("Example 2: Text Query Recommendation")
    print("="*60)
    
    df = load_netflix()
    df = preprocess(df)
    
    recommender = USERecommender()
    recommender.load_embeddings()
    
    # Search by description
    query = "romantic comedy with strong female lead"
    recommendations = recommender.recommend_by_text(
        query,
        df,
        top_n=5
    )
    
    print(f"\nQuery: '{query}'\n")
    print("Results:")
    for idx, row in recommendations.iterrows():
        print(f"{idx+1}. {row['title']}")
        if 'listed_in' in row:
            print(f"   Genres: {row['listed_in']}")
        print()


def example_3_filtered_recommendations():
    """Example 3: Recommendations with filters"""
    print("\n" + "="*60)
    print("Example 3: Filtered Recommendations")
    print("="*60)
    
    df = load_netflix()
    df = preprocess(df)
    
    recommender = USERecommender()
    recommender.load_embeddings()
    recommender.load_similarity_matrix()
    recommender.load_index_map()
    
    # Get recommendations for movies only
    title = "The Dark Knight"
    recommendations = recommender.recommend(
        title,
        df,
        top_n=10,
        exclude_same_type=True  # Only movies
    )
    
    print(f"\nMovie recommendations for '{title}':")
    print(recommendations[['title', 'type', 'release_year', 'similarity_score']])


def example_4_batch_recommendations():
    """Example 4: Get recommendations for multiple titles"""
    print("\n" + "="*60)
    print("Example 4: Batch Recommendations")
    print("="*60)
    
    df = load_netflix()
    df = preprocess(df)
    
    recommender = USERecommender()
    recommender.load_embeddings()
    recommender.load_similarity_matrix()
    recommender.load_index_map()
    
    # Multiple titles
    titles = ["Breaking Bad", "Friends", "The Matrix"]
    
    for title in titles:
        print(f"\n--- Recommendations for '{title}' ---")
        recs = recommender.recommend(title, df, top_n=3)
        for idx, row in recs.iterrows():
            print(f"  {idx+1}. {row['title']}")


def example_5_explore_embeddings():
    """Example 5: Explore embedding space"""
    print("\n" + "="*60)
    print("Example 5: Embedding Analysis")
    print("="*60)
    
    df = load_netflix()
    df = preprocess(df)
    
    recommender = USERecommender()
    embeddings = recommender.load_embeddings()
    
    print(f"\nEmbedding statistics:")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.4f}")
    print(f"Std: {embeddings.std():.4f}")
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")
    
    # Find most similar pair
    sim_matrix = recommender.load_similarity_matrix()
    
    # Exclude diagonal (self-similarity)
    import numpy as np
    sim_matrix_copy = sim_matrix.copy()
    np.fill_diagonal(sim_matrix_copy, 0)
    
    max_sim_idx = np.unravel_index(
        sim_matrix_copy.argmax(), 
        sim_matrix_copy.shape
    )
    
    print(f"\nMost similar pair:")
    print(f"1. {df.iloc[max_sim_idx[0]]['title']}")
    print(f"2. {df.iloc[max_sim_idx[1]]['title']}")
    print(f"Similarity: {sim_matrix[max_sim_idx]:.4f}")


def example_6_custom_pipeline():
    """Example 6: Custom recommendation pipeline"""
    print("\n" + "="*60)
    print("Example 6: Custom Pipeline")
    print("="*60)
    
    # Load and preprocess
    df = load_netflix()
    df = preprocess(df)
    
    # Filter dataset (e.g., only recent content)
    recent_df = df[df['release_year'] >= 2015].copy()
    print(f"\nFiltered to {len(recent_df)} titles from 2015+")
    
    # Build custom recommender
    recommender = USERecommender()
    
    print("\nBuilding embeddings for filtered dataset...")
    embeddings = recommender.build_embeddings(
        recent_df, 
        batch_size=32,
        save=False  # Don't overwrite main models
    )
    
    print("Building similarity matrix...")
    sim_matrix = recommender.build_similarity_matrix(
        embeddings,
        save=False
    )
    
    print("Building index map...")
    idx_map = recommender.build_index_map(recent_df, save=False)
    
    # Get recommendations
    title = "Squid Game"
    recs = recommender.recommend(title, recent_df, top_n=5)
    
    print(f"\nRecent content similar to '{title}':")
    print(recs[['title', 'release_year', 'type']])


def example_7_error_handling():
    """Example 7: Error handling"""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)
    
    df = load_netflix()
    df = preprocess(df)
    
    recommender = USERecommender()
    recommender.load_embeddings()
    recommender.load_similarity_matrix()
    recommender.load_index_map()
    
    # Test with non-existent title
    fake_title = "This Movie Does Not Exist 12345"
    recs = recommender.recommend(fake_title, df)
    
    if recs.empty:
        print(f"✓ Correctly handled non-existent title")
    
    # Test with partial match
    partial_title = "dark"
    recs = recommender.recommend(partial_title, df, top_n=3)
    
    if not recs.empty:
        print(f"\n✓ Found recommendations with partial match:")
        print(recs[['title']])


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_recommendation,
        example_2_text_query,
        example_3_filtered_recommendations,
        example_4_batch_recommendations,
        example_5_explore_embeddings,
        example_6_custom_pipeline,
        example_7_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
        
        input("\nPress Enter to continue to next example...")


if __name__ == "__main__":
    print("Netflix Recommender System - Usage Examples")
    print("\nMake sure you have run 'python main.py' first to build the models!")
    
    print("\nAvailable examples:")
    print("1. Basic recommendation")
    print("2. Text query recommendation")
    print("3. Filtered recommendations")
    print("4. Batch recommendations")
    print("5. Explore embeddings")
    print("6. Custom pipeline")
    print("7. Error handling")
    print("8. Run all examples")
    
    choice = input("\nEnter example number (1-8): ").strip()
    
    examples_map = {
        '1': example_1_basic_recommendation,
        '2': example_2_text_query,
        '3': example_3_filtered_recommendations,
        '4': example_4_batch_recommendations,
        '5': example_5_explore_embeddings,
        '6': example_6_custom_pipeline,
        '7': example_7_error_handling,
        '8': run_all_examples
    }
    
    example_func = examples_map.get(choice)
    if example_func:
        example_func()
    else:
        print("Invalid choice")