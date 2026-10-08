# CNN-Based Waste Classification System

An end-to-end Deep Learning system built using Convolutional Neural Networks (CNN) to categorize waste items into Organic and Recyclable categories.

## Repository Structure
- dataset/: Training and testing image splits
- notebooks/: Analysis notebooks
- src/: Model architecture and prediction scripts
- results/: Output graphs and predictions
- screenshots/: Execution output

## Setup & Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/model.py`
3. Predict an image: `python src/predict.py dataset/test/organic/sample.jpg`