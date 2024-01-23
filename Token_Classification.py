from transformers import BertTokenizer, BertForTokenClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Bert is a powerful NLP model developed by Google."

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Get model predictions
outputs = model(**inputs).logits

# Get predicted labels (class with maximum probability for each token)
predictions = outputs.argmax(dim=2)

# Decode the predicted labels back to human-readable format
predicted_labels = [tokenizer.decode(prediction) for prediction in predictions[0]]

# Print the original sentence and predicted labels
print("Original Sentence:", sentence)
print("Predicted Labels:", predicted_labels)
