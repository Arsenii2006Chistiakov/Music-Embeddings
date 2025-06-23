import torchaudio
import torch
import torchaudio.transforms as T
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import os # Import os for temporary file handling

import numpy as np
from typing import Dict, List, Union, Tuple
import json
from pathlib import Path
from pydub import AudioSegment
import uuid

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

# Assuming 'model' and 'processor' are already defined from the previous cells
# If not, you would need to load them within or before calling this function.

def create_mert_embedding(audio_filepath, timestamp_range, model, processor):
    """
    Generates a MERT embedding for a given audio snippet defined by a timestamp range.

    Args:
        audio_filepath (str): The path to the audio file.
        timestamp_range (str): The time range of the snippet in format "M:SS - M:SS".
        model: The loaded MERT model.
        processor: The loaded MERT feature extractor/processor.

    Returns:
        torch.Tensor: The MERT embedding of the audio snippet.
        None: If there is an error loading the audio or processing the snippet.
    """
    temp_snippet_path = None
    try:
        # Get the audio snippet
        temp_snippet_path = get_snippet_mp3(audio_filepath, timestamp_range)

        # Load the snippet waveform
        waveform, orig_sr = torchaudio.load(temp_snippet_path)

    except Exception as e:
        print(f"Error processing snippet for {audio_filepath} with timestamp {timestamp_range}: {e}")
        return None
    finally:
        # Clean up the temporary snippet file
        if temp_snippet_path and os.path.exists(temp_snippet_path):
            os.remove(temp_snippet_path)


    target_sr = processor.sampling_rate
    if orig_sr != target_sr:
        resampler = T.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ensure waveform is a PyTorch tensor before processing
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform)

    try:
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=target_sr, return_tensors="pt")
    except Exception as e:
        print(f"Error processing waveform for {audio_filepath} snippet: {e}")
        return None

    # Run model inference
    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=True)
        except Exception as e:
            print(f"Error during model inference for {audio_filepath} snippet: {e}")
            return None


    # Get all hidden states (layers, time steps, features)
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)  # [25, T, 1024]

    # Time-mean pool each layer
    time_reduced = all_layer_hidden_states.mean(dim=1)  # [25, 1024]

    # Learnable weighted average (optional) - using a simple mean here as aggregator is not defined in this scope
    # If you have a trained aggregator, you would use it here.
    # aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1) # Assuming aggregator is defined elsewhere
    # weighted_embedding = aggregator(time_reduced.unsqueeze(0)).squeeze()  # [1024]
    weighted_embedding = time_reduced.mean(dim=0) # Using mean as an example if no aggregator is available


    return weighted_embedding


def create_database(annotations_path: str, model, processor):
    """
    Create a database of embeddings from a list of annotations.
    """
    database = {}
    with open(annotations_path, 'r') as f:
        content = f.read()
        start = content.find('[')
        end = content.rfind(']') + 1
        annotations = json.loads(content[start:end])
    for an in annotations:
        song1_full_path = an["song1"]
        song2_full_path = an["song2"]
        timestamp_range1=an["snippet1"]
        timestamp_range2=an["snippet2"]
        embedding1 = create_mert_embedding(song1_full_path, timestamp_range1, model, processor)
        embedding2 = create_mert_embedding(song2_full_path, timestamp_range2, model, processor)
        database[song1_full_path] = embedding1
        database[song2_full_path] = embedding2
    return database


if __name__ == "__main__":
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model.eval()
    audio_filepath = "/Users/arseniichistiakov/Desktop/Python files/audio comparison/music dataset/processed/GIRLS.mp3"
    timestamp_range = "0:12 - 0:24"
    embeddings = create_mert_embedding(audio_filepath, timestamp_range, model, processor)
    print(embeddings)