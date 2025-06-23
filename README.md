# Music Embeddings Evaluation

This project implements a framework for evaluating music embeddings using the MERT model. It focuses on comparing audio snippets and measuring the model's ability to identify similar musical segments.

## Project Structure

- `criteria.py`: Implementation of evaluation metrics and scoring functions
- `get_embeddings.py`: Functions for creating MERT embeddings from audio snippets
- `workflow.py`: Main workflow combining embedding creation and evaluation
- `annotations.txt`: Dataset of annotated song pairs with timestamps

## Features

- Extract audio snippets from specific timestamps
- Generate MERT embeddings for audio segments
- Evaluate embedding quality using cosine similarity
- Compare snippets using top-k matching
- Detailed evaluation metrics and reporting

## Requirements

```
torch
torchaudio
transformers
pydub
numpy
```

## Usage

1. Prepare your annotations file with song pairs and timestamps
2. Run the evaluation:
```python
python workflow.py
```

The script will:
1. Load the MERT model
2. Create embeddings for all snippets
3. Evaluate the embeddings
4. Print detailed results and success rate

## Evaluation Metrics

The evaluation produces:
- Average score across all pairs
- Success rate (percentage of correct matches)
- Detailed results for each query
- Top-k matches with similarity scores 