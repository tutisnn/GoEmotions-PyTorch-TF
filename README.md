# GoEmotions-PyTorch-TF

This project focuses on classifying emotions from text using the **DistilBERT** model, implemented in both **PyTorch** and **TensorFlow**. The dataset used is the "GoEmotions" dataset, which contains a variety of emotional labels.

## Features:
1. **Data Loading and Preprocessing**:
   - Loads the "GoEmotions" dataset using the Hugging Face `datasets` library.
   - Filters the data to remove multi-label entries.
   - Maps labels to a string format for better visualization and understanding.

2. **Data Visualization**:
   - Visualizes class distribution using bar charts to observe any data imbalances.

3. **Model Training**:
   - Implements **DistilBERT** for sequence classification in both **PyTorch** and **TensorFlow** frameworks.
   - Trains the model on the preprocessed dataset, using **accuracy** as the evaluation metric.
   - Includes callbacks to save the model to the Hugging Face Hub and to monitor metrics during training.

4. **Evaluation**:
   - Evaluates the trained models on the validation set and displays a **Confusion Matrix** to visualize model performance across different emotions.
   
5. **Emotion Prediction**:
   - The model can classify user input into predefined emotion categories using a text classification pipeline.
   - A command-line interface allows users to interact with the model, input sentences, and receive emotion predictions.

You can use these models for emotion classification in your own projects.
### PyTorch Model:
To use the PyTorch model for emotion classification, you can use the following code snippet:

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="tuhanasinan/go-emotions-distilbert-pytorch")
```
### TensorFlow Model:
To use the TensorFlow model for emotion classification, you can use the following code snippet:

```python

from transformers import pipeline

pipe = pipeline("text-classification", model="tuhanasinan/go_emotions-distilbert-tensorflow")
```
## Link

Here are the links to the dataset and pre-trained and fine-tuned models used in this project:

- **Dataset**: [GoEmotions Dataset](https://huggingface.co/datasets/google_research_datasets/go_emotions)
- **Distilbert Model**:[DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)
- **PyTorch Model**: [go-emotions-distilbert-pytorch](https://huggingface.co/tuhanasinan/go-emotions-distilbert-pytorch)
- **TensorFlow Model**: [go-emotions-distilbert-tensorflow](https://huggingface.co/tuhanasinan/go-emotions-distilbert-tensorflow)

Kaggle Links
[Go_Emotions_Analaysis_DistilBERT_pytorch](https://www.kaggle.com/code/tuhanasinan/go-emotions-analysis-pytorch)




