# AI Image Caption Generator

## Overview

This project generates descriptive captions for images using Deep Learning and Computer Vision techniques. The model analyzes an input image and produces a natural language description of its contents.

## Features

* Upload an image and generate captions automatically.
* Uses CNN for image feature extraction.
* Uses LSTM/Transformer for caption generation.
* Supports custom image datasets.
* Easy-to-use interface.

## Technologies Used

* Python
* TensorFlow 
* NumPy
* Pandas
* NLTK
* Streamlit (Optional for UI)

## Project Structure

```
AI-Image-Captioning/
│
├── dataset/
├── models/
├── notebooks/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd AI-Image-Captioning
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python train.py
```

### Generate Caption

```bash
python predict.py --image image.jpg
```

## How It Works

1. Input image is provided.
2. CNN extracts image features.
3. Features are passed to the language model.
4. The model generates a meaningful caption.
5. Caption is displayed to the user.

## Example

**Input Image:** Dog playing with a ball

**Generated Caption:**
"A brown dog is playing with a ball in the grass."

## Future Enhancements

* Transformer-based caption generation
* Attention mechanisms
* Multi-language captions
* Real-time webcam captioning

## Author

Ankit Kashyap


