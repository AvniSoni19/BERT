#BertTokenizer import which tokenizes the sentences
from transformers import BertTokenizer

# Load pre_trained BERT Tokenizer 
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#Input Text
text = "We are excited about the possibility of having you join the Meesho team and look forward to your participation in the next phase of the selection process. Wish you all the best."

# Tokenize and encode the text
encoding = tokenizer.encode(text)  # By default it will be encoded in the form of id

# Print the tokenized id
print("Token IDs: ", encoding)

# Convert token ids into token
tokens = tokenizer.convert_ids_to_tokens(encoding)

# Print the tokens
print("Tokens: ",tokens)
