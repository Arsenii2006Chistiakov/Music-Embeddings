import os
import json
import numpy as np
import essentia
import essentia.standard as es
from essentia import Pool
from collections import defaultdict

# --- 1. Configuration ---
MODEL_PATH = "models/vggish-audioset-10.pb"
JSON_FILE = "song_pairs.json"


# --- 2. Core Functions ---

def get_full_song_embedding(audio_path, model):
    """
    Loads an ENTIRE audio file and computes its embedding.
    This reuses the pre-loaded model for efficiency.
    """
    if not os.path.exists(audio_path):
        print(f"  [Warning] File not found: {audio_path}")
        return None
        
    try:
        # Load the entire audio file. We downsample to 16kHz as required by the model.
        audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
        
        if np.max(np.abs(audio)) < 1e-5: # Check for silence
            print(f"  [Warning] Audio is silent, skipping: {os.path.basename(audio_path)}")
            return None

        # The VGGish model expects specific inputs. We process the audio in chunks.
        spec_transform = es.SymmetricMelBands(sampleRate=16000)
        log_transform = es.Log()
        pool = Pool()
        
        # Process audio in non-overlapping 0.96-second frames (as per VGGish standard)
        for frame in es.FrameGenerator(audio, frameSize=15360, hopSize=15360, startFromZero=True):
            if np.max(np.abs(frame)) > 0:
                mel_spec = spec_transform(frame)
                log_mel_spec = log_transform(mel_spec)
                embedding = model(log_mel_spec)
                pool.add('embedding', embedding)

        if 'embedding' not in pool.descriptorNames():
            return None

        # Aggregate the embeddings of all frames by taking the mean
        return np.mean(pool['embedding'], axis=0)

    except Exception as e:
        print(f"  [Error] Could not process {os.path.basename(audio_path)}: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Computes the cosine similarity between two vectors."""
    if vec1 is None or vec2 is None: return 0
    vec1_norm, vec2_norm = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if vec1_norm == 0 or vec2_norm == 0: return 0
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)


# --- 3. Main Execution Logic ---

def main():
    # --- Part 1: Load Ground Truth and All Unique Songs ---
    print("Step 1: Loading ground truth pairs and cataloging songs...")
    try:
        with open(JSON_FILE, 'r') as f:
            ground_truth_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Ground truth file not found at '{JSON_FILE}'")
        return

    # Use a set of frozensets for easy, order-independent lookup of correct pairs
    ground_truth_pairs = set()
    all_songs = set()
    for pair in ground_truth_data:
        song1, song2 = pair['song1'], pair['song2']
        ground_truth_pairs.add(frozenset([song1, song2]))
        all_songs.add(song1)
        all_songs.add(song2)

    all_songs = sorted(list(all_songs)) # Convert to a sorted list
    print(f"Found {len(ground_truth_pairs)} ground truth pairs and {len(all_songs)} unique songs.")
    print("-" * 50)

    # --- Part 2: Generate and Cache Embeddings for All Songs ---
    print("Step 2: Generating embeddings for all songs... (This may take a while)")
    
    # Load the model once to be reused
    try:
        model = es.TensorflowPredictVGGish(graphFilename=MODEL_PATH)
    except Exception as e:
        print(f"FATAL: Could not load the model. Ensure it's downloaded. Error: {e}")
        return

    embeddings_cache = {}
    for song_path in all_songs:
        print(f"  Processing: {os.path.basename(song_path)}")
        embedding = get_full_song_embedding(song_path, model)
        if embedding is not None:
            embeddings_cache[song_path] = embedding
    
    # Remove songs that failed to process from our list
    all_songs = sorted(list(embeddings_cache.keys()))
    print(f"Successfully generated embeddings for {len(all_songs)} songs.")
    print("-" * 50)

    # --- Part 3: Find the Best Match for Each Song ---
    print("Step 3: Finding the best match for each song...")
    predicted_matches = {}
    for i, song_a in enumerate(all_songs):
        best_match_song = None
        highest_similarity = -1

        for j, song_b in enumerate(all_songs):
            if i == j:  # Don't compare a song to itself
                continue
            
            similarity = cosine_similarity(embeddings_cache[song_a], embeddings_cache[song_b])

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_song = song_b
        
        predicted_matches[song_a] = best_match_song
    print("Finished finding all best matches.")
    print("-" * 50)

    # --- Part 4: Evaluate Accuracy ---
    print("Step 4: Evaluating accuracy against ground truth...")
    correct_predictions = 0
    total_predictions = len(predicted_matches)
    
    # Display each prediction and check if it's correct
    for song, predicted_partner in predicted_matches.items():
        
        # Create a frozenset for the predicted pair to check against the ground truth
        predicted_pair_set = frozenset([song, predicted_partner])
        
        is_correct = "INCORRECT"
        if predicted_pair_set in ground_truth_pairs:
            correct_predictions += 1
            is_correct = "CORRECT"

        print(f"  - For '{os.path.basename(song)}', best match is '{os.path.basename(predicted_partner)}' -> [{is_correct}]")

    print("-" * 50)
    
    # --- Final Results ---
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    print("--- EVALUATION SUMMARY ---")
    print(f"Correctly Identified Matches: {correct_predictions}")
    print(f"Total Songs Evaluated: {total_predictions}")
    print(f"Algorithm Accuracy: {accuracy:.2f}%")
    print("--------------------------")
    
if __name__ == '__main__':
    main()