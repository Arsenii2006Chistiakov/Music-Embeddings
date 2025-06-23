#criteria for the audio comparison

import numpy as np
from typing import Dict, List, Union, Tuple
import json
from pathlib import Path
from pydub import AudioSegment
import uuid

#parse timestamp works

def parse_timestamp(timestamp: str) -> float:
    """
    Convert timestamp from format "M:SS" to seconds
    Example: "1:23" -> 83.0 seconds
    """
    # Remove any whitespace and split by '-' to handle ranges
    if '-' in timestamp:
        time = timestamp.split('-')[0].strip()
    else:
        time = timestamp.strip()
        
    # Split into minutes and seconds
    minutes, seconds = map(int, time.split(':'))
    return minutes * 60 + seconds



#get snippet works


def get_snippet_mp3(audio_path: str, timestamp_range: str) -> str:
    """
    Extract audio snippet from given time range
    Args:
        audio_path: Path to audio file
        timestamp_range: Time range in format "M:SS - M:SS"
    Returns:
        Path to temporary file with extracted snippet
    """
    # Split timestamp range into start and end
    start_time, end_time = map(str.strip, timestamp_range.split('-'))
    
    # Convert to seconds
    start_seconds = parse_timestamp(start_time)
    end_seconds = parse_timestamp(end_time)
    
    # Load audio file
    audio = AudioSegment.from_mp3(audio_path)
    
    # Convert seconds to milliseconds
    start_ms = int(start_seconds * 1000)
    end_ms = int(end_seconds * 1000)
    
    # Extract snippet
    snippet = audio[start_ms:end_ms]
    
    # Save snippet to temporary file
    temp_path = f"temp_snippet_{uuid.uuid4()}.mp3"
    snippet.export(temp_path, format="mp3")
    
    return temp_path

# needs to be checked

def compute_score(query_filename: str,
                 target_filename: str, 
                 database: Dict[str, np.ndarray],
                 k: int = 3) -> float:
    """
    Compute the score for a query file against a database of embeddings.
    Returns 1 if the target is in top k matches, 0 otherwise.
    
    Args:
        query_filename: Name/path of the query audio file
        target_filename: Name/path of the target audio file to look for in top k
        database: Dictionary mapping filenames to their embeddings
        k: Number of top matches to consider (default 3)
    
    Returns:
        float: 1.0 if target in top k, 0.0 otherwise
    """
    # Get query embedding from database
    if query_filename not in database:
        raise KeyError(f"Query file {query_filename} not found in database")
    query_embedding = database[query_filename]
    
    # Get all embeddings and filenames
    filenames = list(database.keys())
    embeddings = np.array([database[fname] for fname in filenames])
    
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norms = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # Compute cosine similarity (1 - cosine distance)
    similarities = np.dot(db_norms, query_norm)
    
    # Apply softmax to get probability scores
    exp_similarities = np.exp(similarities)
    softmax_scores = exp_similarities / np.sum(exp_similarities)
    
    # Get top k indices
    top_k_indices = np.argsort(softmax_scores)[-k:]
    
    # Get top k filenames
    top_k_files = [filenames[i] for i in top_k_indices]
    
    # Check if target is in top k
    return 1.0 if target_filename in top_k_files else 0.0


# needs to be checked

def evaluate_embeddings(database: Dict[str, np.ndarray], 
                       annotations_path: str,
                       k: int = 3) -> Tuple[float, List[Dict]]:
    """
    Evaluate embedding quality using annotation pairs.
    
    Args:
        database: Dictionary mapping filenames to their embeddings
        annotations_path: Path to annotations file with song pairs
        k: Number of top matches to consider (default 3)
    
    Returns:
        Tuple containing:
        - float: Average score across all pairs (0.0 to 1.0)
        - List[Dict]: Detailed results for each pair
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        content = f.read()
        start = content.find('[')
        end = content.rfind(']') + 1
        annotations = json.loads(content[start:end])
    
    results = []
    total_score = 0
    n_pairs = len(annotations)
    
    for pair in annotations:
        query_file = pair['song1']
        target_file = pair['song2']
        
        try:
            # Compute score for this pair
            score = compute_score(
                query_filename=query_file,
                target_filename=target_file,
                database=database,
                k=k
            )
            
            # Get similarity rankings for analysis
            similarities = print_similarity_scores(query_file, database, k)
            
            result = {
                'query_file': query_file,
                'query_snippet': pair['snippet1'],
                'target_file': target_file,
                'target_snippet': pair['snippet2'],
                'score': score,
                'top_k_matches': similarities
            }
            
            results.append(result)
            total_score += score
            
        except KeyError as e:
            print(f"Warning: Skipping pair due to missing file in database: {e}")
            n_pairs -= 1  # Adjust count for missing pairs
    
    # Compute average score
    average_score = total_score / n_pairs if n_pairs > 0 else 0.0
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total pairs evaluated: {n_pairs}")
    print(f"Average score: {average_score:.3f}")
    print(f"Success rate: {average_score * 100:.1f}%")
    print(f"\nDetailed Results:")
    
    for result in results:
        print(f"\nQuery: {Path(result['query_file']).stem} ({result['query_snippet']})")
        print(f"Target: {Path(result['target_file']).stem} ({result['target_snippet']})")
        print(f"Score: {result['score']}")
        print("Top matches:")
        for fname, sim_score in result['top_k_matches']:
            print(f"  {Path(fname).stem}: {sim_score:.3f}")
    
    return average_score, results


# needs to be checked


def print_similarity_scores(query_filename: str,
                          database: Dict[str, np.ndarray],
                          k: int = 3) -> List[tuple]:
    """
    Helper function to print similarity scores for debugging/analysis.
    Returns list of (filename, score) tuples for top k matches.
    """
    if query_filename not in database:
        raise KeyError(f"Query file {query_filename} not found in database")
    
    query_embedding = database[query_filename]
    filenames = list(database.keys())
    embeddings = np.array([database[fname] for fname in filenames])
    
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norms = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # Compute similarities
    similarities = np.dot(db_norms, query_norm)
    exp_similarities = np.exp(similarities)
    softmax_scores = exp_similarities / np.sum(exp_similarities)
    
    # Get top k matches
    top_k_indices = np.argsort(softmax_scores)[-k:][::-1]  # Reverse to get descending order
    top_k_matches = [(filenames[i], softmax_scores[i]) for i in top_k_indices]
    
    return top_k_matches

if __name__ == "__main__":
    annotations_path = "/Users/arseniichistiakov/Desktop/Python files/audio comparison/annotations.txt"
    with open(annotations_path, 'r') as f:
        content = f.read()
        start = content.find('[')
        end = content.rfind(']') + 1
        annotations = json.loads(content[start:end])
    
    # Test timestamp parsing
    for annotation in annotations:
        print(f"\nProcessing {Path(annotation['song1']).stem}")
        print(f"Snippet 1: {annotation['snippet1']}")
        start_seconds = parse_timestamp(annotation['snippet1'])
        end_seconds = parse_timestamp(annotation['snippet1'].split('-')[1])
        print(f"Converted to seconds: {start_seconds} - {end_seconds}")
        
        # Extract snippet
        try:
            snippet_path = get_snippet_mp3(annotation['song1'], annotation['snippet1'])
            print(f"Snippet saved to: {snippet_path}")
        except Exception as e:
            print(f"Error extracting snippet: {str(e)}")
        break  # Just test the first one


