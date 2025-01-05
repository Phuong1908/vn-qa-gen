# data_agumentor/config.py
"""
Configuration and constants for the Vietnamese QA generation
"""

# Question type configurations
QUESTION_TYPES = [
    "WHO", "WHERE", "WHEN", "WHY", "WHAT", "HOW", "OTHER"
]

QUESTION_PATTERNS = {
    "Who": ["ai", "người nào"],
    "Where": ["ở đâu", "nơi nào", "đâu"],
    "When": ["khi nào", "lúc nào", "thời gian nào", "thời điểm nào"],
    "Why": ["tại sao", "vì sao", "do đâu"],
    "How many": ["bao nhiêu"],
    "How": ["thế nào", "như thế nào"],
    "What": ["là gì", "gì", "nào"],
    "OTHER": ["other"]
}

# Vietnamese specific configurations
STOP_WORDS = ['của', 'và', 'các', 'những', 'một', 'với', 'là']
INVALID_START_WORDS = ['của', 'và', 'hay', 'hoặc', 'với', 'bởi', 'theo']
VALID_CHUNK_LABELS = ['NP', 'VP', 'PP', 'AP']
INVALID_POS_TAGS = ['E', 'C', 'T', 'Y', 'B']

# data_agumentor/constants.py
NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE = [
    'của', 'và', 'hay', 'hoặc', 'với', 'bởi', 'theo',
    'là', 'được', 'bị', ',', '?', ';', '!', '.'
]

FUNCTION_WORDS_LIST = [
    'của', 'và', 'hay', 'hoặc', 'với', 'bởi', 'theo',
    'là', 'được', 'bị', 'trong', 'ngoài', 'trên', 'dưới',
    'về', 'từ', 'đến', 'tại', 'trong', 'một', 'các', 'những'
]

QUESTION_TYPES = [
    "WHO", "WHERE", "WHEN", "WHY", "WHAT", "HOW", "OTHER"
]

# Other configurations
MAX_ANSWER_LENGTH = 30
MAX_CLUE_DISTANCE = 20
DISTANCE_THRESHOLD = 5
