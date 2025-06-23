from get_embeddings import create_mert_embedding, create_database
from criteria import evaluate_embeddings
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch

def evaluate_mert_embeddings(annotations_path: str, k: int = 3):
    """
    Complete workflow to evaluate MERT embeddings on the music dataset:
    1. Load MERT model
    2. Create database of embeddings for all snippets
    3. Evaluate the embeddings using the criteria
    """
    print("Loading MERT model...")
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model.eval()

    print("\nCreating database of embeddings...")
    database = create_database(annotations_path, model, processor)
    print(f"Created embeddings for {len(database)} audio snippets")

    print("\nEvaluating embeddings...")
    average_score, detailed_results = evaluate_embeddings(
        database=database,
        annotations_path=annotations_path,
        k=k
    )

    return average_score, detailed_results

if __name__ == "__main__":
    annotations_path = "/Users/arseniichistiakov/Desktop/Python files/audio comparison/annotations.txt"
    k = 3  # Number of top matches to consider
    
    print("Starting MERT embeddings evaluation workflow...")
    average_score, results = evaluate_mert_embeddings(annotations_path, k)
    
    print("\nFinal Results:")
    print(f"Average Score: {average_score:.3f}")
    print(f"Success Rate: {average_score * 100:.1f}%")








