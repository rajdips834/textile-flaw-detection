# Textile Flaw Detection System

A Python-based system designed to automatically detect and classify flaws in textile fabrics using image processing and machine learning techniques. This project aims to improve quality control efficiency and accuracy in textile production by automating the defect detection process.

## Features

- Detect various types of textile flaws such as color defects, cuts, holes, thread issues, and metal contamination.
- Preprocessing of fabric images for enhanced defect visibility.
- Uses computer vision techniques and neural networks for classification.
- User-friendly interface for uploading fabric images and visualizing detected defects.
- High accuracy and real-time detection capabilities.

## Installation

1. Clone this repository:
git clone <repository-url>

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. Install required packages:
pip install -r requirements.txt

## Dataset

The system uses a dataset containing fabric images with labeled defect types such as:
- Good (no defect)
- Color defect
- Cut
- Hole
- Thread defect
- Metal contamination

Images are typically preprocessed and resized for consistent input dimensions.

## Technologies Used

- Python 3.x
- OpenCV for image processing
- TensorFlow/Keras or PyTorch for machine learning model training and inference
- NumPy and Pandas for data handling
- Matplotlib or similar for visualization

## How It Works

1. Image preprocessing to normalize lighting and enhance defect characteristics.
2. Feature extraction using texture analysis and color histograms.
3. Machine learning model classifies the defects based on extracted features.
4. Detected defects are visualized on the fabric image for easy inspection.

## Evaluation

The system is evaluated using metrics such as accuracy, precision, recall, and F1-score on a test dataset to ensure reliable performance.
