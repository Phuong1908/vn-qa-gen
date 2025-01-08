# Add any webapp-specific configurations here
# webapp/config.py
import os

# Model path - adjust this to point to your model
MODEL_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "output/bartpho-syllable")
MAX_PARAGRAPH_LENGTH = 1000
DEFAULT_QUESTION_STYLE = 'simple'
