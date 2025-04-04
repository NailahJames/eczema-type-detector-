# Eczema Type Classifier

This is a PyTorch model that classifies skin condition images into one of four types of eczema:

- Atopic
- Contact
- Dyshidrotic
- Nummular

## Input
- An image of an eczema-affected area (`.jpg`, `.png`)

## Output
- The predicted eczema type as a label (e.g., `"atopic"`)

## How It Works
- This model uses a simple CNN trained in Google Colab on a custom dataset.
- The model is downloaded from Google Drive at runtime.

## Dependencies
- torch
- torchvision
- Pillow
- requests
