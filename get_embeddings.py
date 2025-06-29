import torchaudio
import torch
import torchaudio.transforms as T
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import os # Import os for temporary file handling

import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Callable
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

def create_mert_embedding(
    audio_filepath: str, 
    timestamp_range: str, 
    model: AutoModel, 
    processor: Wav2Vec2FeatureExtractor,
    aggregator: Optional[Union[nn.Module, Callable]] = None,
    aggregator_kwargs: Optional[Dict] = None
) -> torch.Tensor:
    """
    Generates a MERT embedding for a given audio snippet defined by a timestamp range.

    Args:
        audio_filepath (str): The path to the audio file.
        timestamp_range (str): The time range of the snippet in format "M:SS - M:SS".
        model: The loaded MERT model.
        processor: The loaded MERT feature extractor/processor.
        aggregator: Optional aggregator to combine hidden states. Can be:
            - nn.Module: A PyTorch module that processes the hidden states
            - Callable: A function that takes hidden states and returns embedding
            - None: Uses mean pooling (default)
        aggregator_kwargs: Optional dictionary of keyword arguments for the aggregator

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

    # Apply the aggregator if provided
    if aggregator is not None:
        try:
            if isinstance(aggregator, nn.Module):
                # If it's a PyTorch module
                if aggregator_kwargs:
                    weighted_embedding = aggregator(time_reduced, **aggregator_kwargs)
                else:
                    weighted_embedding = aggregator(time_reduced)
            elif callable(aggregator):
                # If it's a function
                if aggregator_kwargs:
                    weighted_embedding = aggregator(time_reduced, **aggregator_kwargs)
                else:
                    weighted_embedding = aggregator(time_reduced)
            else:
                raise ValueError("Aggregator must be either nn.Module or callable")
        except Exception as e:
            print(f"Error applying aggregator: {e}")
            print("Falling back to mean pooling")
            weighted_embedding = time_reduced.mean(dim=0)
    else:
        # Default to mean pooling if no aggregator is provided
        weighted_embedding = time_reduced.mean(dim=0)

    return weighted_embedding

# Example aggregator classes
class AttentionAggregator(nn.Module):
    """
    Attention-based aggregator that learns to weight different layers.
    """
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: [num_layers, hidden_size]
        attention_weights = torch.softmax(self.attention(hidden_states), dim=0)  # [num_layers, 1]
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=0)  # [hidden_size]
        return weighted_sum

class ConvAggregator(nn.Module):
    """
    Convolutional aggregator that applies 1D convolution across layers.
    """
    def __init__(self, hidden_size: int = 1024, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=25, out_channels=out_channels, kernel_size=1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: [num_layers, hidden_size]
        output = self.conv(hidden_states.unsqueeze(0))  # [1, out_channels, hidden_size]
        return output.squeeze()  # [hidden_size]
    


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

class HeuristicAggregator(nn.Module):
    """
    Heuristic aggregator that applies different weighting schemes to combine layers.
    Supports linear, exponential, and logarithmic growth/decay patterns.
    
    Args:
        num_layers: Number of layers to aggregate (default: 25 for MERT)
        weight_type: Type of weighting scheme to use:
            - 'linear_up': Linear growth (more weight to top layers)
            - 'linear_down': Linear decay (more weight to bottom layers)
            - 'exp_up': Exponential growth
            - 'exp_down': Exponential decay
            - 'log_up': Logarithmic growth
            - 'log_down': Logarithmic decay
        base: Base for exponential weighting (default: 2)
        eps: Small value to avoid log(0) (default: 1e-10)
    """
    def __init__(
        self,
        num_layers: int = 25,
        weight_type: str = 'linear_up',
        base: float = 2.0,
        eps: float = 1e-10
    ):
        super().__init__()
        self.num_layers = num_layers
        self.weight_type = weight_type
        self.base = base
        self.eps = eps
        
        # Register weights as buffer (not parameters since they're fixed)
        self.register_buffer('weights', self._create_weights())
        
    def _create_weights(self) -> torch.Tensor:
        """Creates weights according to specified scheme"""
        if self.weight_type == 'linear_up':
            # Linear growth: [0, 1]
            weights = torch.linspace(0.0, 1.0, self.num_layers)
            
        elif self.weight_type == 'linear_down':
            # Linear decay: [1, 0]
            weights = torch.linspace(1.0, 0.0, self.num_layers)
            
        elif self.weight_type == 'exp_up':
            # Exponential growth: [1, base^(n-1)]
            weights = torch.tensor([self.base ** i for i in range(self.num_layers)])
            
        elif self.weight_type == 'exp_down':
            # Exponential decay: [base^(n-1), 1]
            weights = torch.tensor([self.base ** (self.num_layers - 1 - i) for i in range(self.num_layers)])
            
        elif self.weight_type == 'log_up':
            # Logarithmic growth
            x = torch.linspace(self.eps, 1.0, self.num_layers)
            weights = torch.log(x + 1.0)
            
        elif self.weight_type == 'log_down':
            # Logarithmic decay
            x = torch.linspace(1.0, self.eps, self.num_layers)
            weights = torch.log(x + 1.0)
            
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        return weights
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply weighted combination to hidden states
        
        Args:
            hidden_states: Tensor of shape [num_layers, hidden_size]
            
        Returns:
            Weighted combination of hidden states [hidden_size]
        """
        # Apply weights along the layer dimension
        weighted = hidden_states * self.weights.unsqueeze(1)
        # Sum across layers
        return weighted.sum(dim=0)

    def get_weights(self) -> torch.Tensor:
        """Returns the current weights for inspection"""
        return self.weights

if __name__ == "__main__":
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model.eval()
    
    # Create examples of different heuristic aggregators
    linear_up = HeuristicAggregator(weight_type='linear_up')
    linear_down = HeuristicAggregator(weight_type='linear_down')
    exp_up = HeuristicAggregator(weight_type='exp_up', base=2.0)
    exp_down = HeuristicAggregator(weight_type='exp_down', base=2.0)
    log_up = HeuristicAggregator(weight_type='log_up')
    log_down = HeuristicAggregator(weight_type='log_down')
    
    # Example with conv aggregator
    conv_aggregator = ConvAggregator()
    
    audio_filepath = "/Users/arseniichistiakov/Desktop/Python files/audio comparison/music dataset/processed/GIRLS.mp3"
    timestamp_range = "0:12 - 0:24"
    
    # Get embeddings with different weighting schemes
    embedding_linear_up = create_mert_embedding(
        audio_filepath, 
        timestamp_range, 
        model, 
        processor,
        aggregator=linear_up
    )
    

