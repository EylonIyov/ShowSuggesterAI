import pickle
from pathlib import Path
import sys
import csv
import dotenv
from fuzzywuzzy import fuzz
import numpy as np
from numpy.linalg import norm
from usearch.index import Index


dotenv.load_dotenv()

# Add Modules directory to path
sys.path.append(str(Path(__file__).parent.parent / "Modules"))
from embeddings import build_description_embeddings
from llm_generator import generate_creative_shows, generate_show_images, open_generated_images


def create_embeddings_pickle(csv_path: str, pickle_path: str):
    """Build embeddings from CSV and save to pickle file."""
    csv_path_obj = Path(csv_path)
    description_embeded = build_description_embeddings(csv_path_obj)
    
    # Save embeddings to pickle file
    pickle_path_obj = Path(pickle_path)
    with pickle_path_obj.open("wb") as f:
        pickle.dump(description_embeded, f)
    


def load_embeddings_from_pickle(pickle_path: str) -> dict:
    """Load embeddings dictionary from pickle file.
    
    Returns:
        Dictionary mapping show titles to tuples of (description, embedding vector).
    
    Raises:
        FileNotFoundError: If pickle file doesn't exist.
    """
    pickle_path_obj = Path(pickle_path)
    
    if not pickle_path_obj.exists():
        raise FileNotFoundError(
            f"Pickle file not found: {pickle_path_obj}. "
            "Run create_embeddings_pickle() first to generate it."
        )
    
    with pickle_path_obj.open("rb") as f:
        description_embeded = pickle.load(f)
    
    return description_embeded


def build_usearch_index(embeddings_dict: dict) -> tuple:
    """Build a usearch index from embeddings for fast similarity search.
    
    Args:
        embeddings_dict: Dictionary mapping show titles to (description, embedding).
    
    Returns:
        Tuple of (index, show_titles_list) where show_titles_list maps index positions to titles.
    """
    # Get dimension from first embedding
    first_embedding = next(iter(embeddings_dict.values()))[1]
    dimensions = len(first_embedding)
    
    # Create index with cosine metric
    index = Index(ndim=dimensions, metric='cos')
    
    # Build list to map index positions to show titles
    show_titles = []
    
    # Add all embeddings to the index
    for idx, (show_title, (description, embedding_vector)) in enumerate(embeddings_dict.items()):
        index.add(idx, np.array(embedding_vector, dtype=np.float32))
        show_titles.append(show_title)
    
    return index, show_titles


def load_show_titles_from_csv(csv_path: str) -> list:
    """Load all TV show titles from CSV file.
    
    Returns:
        List of show title strings.
    """
    csv_path_obj = Path(csv_path)
    titles = []
    
    with csv_path_obj.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "").strip()
            if title:
                titles.append(title)
    
    return titles


def find_best_matching_shows(user_shows: list, all_titles: list) -> list:
    """Find best matching show titles using fuzzy matching.
    
    Args:
        user_shows: List of show names entered by user.
        all_titles: List of all available show titles from CSV.
    
    Returns:
        List of best matching show titles.
    """
    matched_shows = []
    
    for user_show in user_shows:
        best_match = None
        best_score = 0
        
        for title in all_titles:
            score = fuzz.ratio(user_show.lower(), title.lower())
            if score > best_score:
                best_score = score
                best_match = title
        
        if best_match and best_score > 60:  # Threshold for acceptable match
            matched_shows.append(best_match)
        
    
    return matched_shows


def get_embeddings_for_shows(matched_shows: list, embeddings_dict: dict) -> list:
    """Extract embedding vectors for the matched shows.
    
    Args:
        matched_shows: List of show titles.
        embeddings_dict: Dictionary mapping show titles to (description, embedding).
    
    Returns:
        List of embedding vectors (numpy arrays).
    """
    embeddings = []
    for show in matched_shows:
        if show in embeddings_dict:
            _, embedding_vector = embeddings_dict[show]
            embeddings.append(np.array(embedding_vector))
    
    return embeddings


def calculate_average_vector(embedding_vectors: list) -> np.ndarray:
    """Calculate the average of multiple embedding vectors.
    
    Args:
        embedding_vectors: List of numpy arrays.
    
    Returns:
        Average vector as numpy array.
    """
    if not embedding_vectors:
        raise ValueError("No embedding vectors provided")
    
    return np.mean(embedding_vectors, axis=0)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
    
    Returns:
        Cosine similarity score (higher is more similar).
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def find_similar_shows_with_index(
    average_vector: np.ndarray,
    index: Index,
    show_titles: list,
    exclude_shows: list,
    top_n: int = 5
) -> list:
    """Find shows most similar to the average vector using usearch index.
    
    Args:
        average_vector: The average embedding vector of user's favorite shows.
        index: Usearch index containing all show embeddings.
        show_titles: List mapping index positions to show titles.
        exclude_shows: List of show titles to exclude (user's input shows).
        top_n: Number of recommendations to return.
    
    Returns:
        List of tuples: (show_title, similarity_score), sorted by similarity (highest first).
    """
    # Search for more results than needed to account for excluded shows
    search_count = top_n + len(exclude_shows) + 10
    
    # Query the index for nearest neighbors
    matches = index.search(average_vector.astype(np.float32), search_count)
    
    # Process results and filter out excluded shows
    recommendations = []
    for idx, distance in zip(matches.keys, matches.distances):
        show_title = show_titles[idx]
        if show_title not in exclude_shows:
            # Convert distance to similarity (usearch returns distances, not similarities)
            # For cosine distance: similarity = 1 - distance
            similarity = 1 - distance
            recommendations.append((show_title, similarity))
            
            if len(recommendations) >= top_n:
                break
    
    return recommendations


def main():#Appllication entry point
    csv_path = str(Path(__file__).parent.parent / "imdb_tvshows - imdb_tvshows.csv")
    pickle_path = str(Path(__file__).parent.parent / "description_embeded.pkl")
    
    # Load embeddings from pickle file
    try:
        description_embeded = load_embeddings_from_pickle(pickle_path)
    except FileNotFoundError:
        create_embeddings_pickle(csv_path, pickle_path)
        description_embeded = load_embeddings_from_pickle(pickle_path)

    # Load all show titles from CSV
    all_titles = load_show_titles_from_csv(csv_path)

    favorite_shows = input("""Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show”:""")
    shows_list = [show.strip() for show in favorite_shows.split(",") if show.strip()]
    
    # Find best matching shows using fuzzy matching
    matched_shows = find_best_matching_shows(shows_list, all_titles)
    while True:
        answer_binary = input(f"\nMaking sure, do you mean {', '.join(matched_shows)} (y/n)")
        if answer_binary.lower() == 'y':
            print(f"\nGreat! You selected: {', '.join(matched_shows)}")
            break
        elif answer_binary.lower() == 'n':
            favorite_shows = input("Sorry about that. Lets try again, please make sure to write the names ofthe tv shows correctly")
            shows_list = [show.strip() for show in favorite_shows.split(",") if show.strip()]
            matched_shows = find_best_matching_shows(shows_list, all_titles)
        else:
            print("Please answer with 'y' or 'n'.")
    
    print("Great! Building search index and generating recommendations…")
    
    # Build usearch index for fast similarity search
    search_index, show_titles = build_usearch_index(description_embeded)
    
    # Extract embedding vectors for the matched shows
    user_show_embeddings = get_embeddings_for_shows(matched_shows, description_embeded)
    
    if not user_show_embeddings:
        print("Error: Could not find embeddings for any of the matched shows.")
        return
    
    # Calculate the average vector of user's favorite shows
    average_vector = calculate_average_vector(user_show_embeddings)
    
    # Find the 5 most similar shows using usearch index
    recommendations = find_similar_shows_with_index(
        average_vector,
        search_index,
        show_titles,
        exclude_shows=matched_shows,
        top_n=3
    )
    
    # Display recommendations to the user
    print("Here are the tv shows that i think you would love:")
    for i, (show_title, similarity_score) in enumerate(recommendations, 1):
        similarity_percent = similarity_score * 135
        print(f"{i}. {show_title} (similarity: {similarity_percent:.1f}%)")
    
    # Get descriptions for the matched shows
    matched_show_descriptions = [
        description_embeded[show][0] for show in matched_shows
    ]
    
    # Generate creative shows using LLM
    print("\nGenerating creative show recommendations...")
    creative_output, shows_data = generate_creative_shows(
        matched_shows,
        matched_show_descriptions,
        recommendations
    )
    
    print(f"\n{creative_output}")
    
    # Generate images for the creative shows
    print("\nGenerating promotional images for the shows...")
    try:
        image_paths = generate_show_images(shows_data)
        print(f"\nShow #1 image saved to: {image_paths.get('show1_image_path', 'Generation failed')}")
        print(f"Show #2 image saved to: {image_paths.get('show2_image_path', 'Generation failed')}")
        
        # Open the generated images
        print("\nOpening generated images...")
        open_generated_images(image_paths)
    except RuntimeError as e:
        print(f"Could not generate images: {str(e)}")
        
            
if __name__ == "__main__":
    main()
    