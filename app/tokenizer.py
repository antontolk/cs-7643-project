import re

def tokenizer_by_word(text):
    # convert to lowercase and split based on words
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

