import re
import emoji

def preprocess_sarcasm(text):
    """
    Applies regex, demojizing, and cleaning.
    """
    if not isinstance(text, str): return ""
    text = re.sub(r"@[^\s]+", "@user", text)
    text = re.sub(r"http\S+", "http", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r"#sarcasm|#sarcastic|#irony|#not", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())