# data_agumentor/config.py
"""
Configuration and constants for the Vietnamese QA generation
"""

# Question type configurations
QUESTION_TYPES = [
    "WHO", "WHERE", "WHEN", "WHY", "WHAT", "HOW", "OTHER"
]

QUESTION_PATTERNS = {
    "WHO": ["ai", "người nào", "ai là"],
    "WHERE": ["ở đâu", "nơi nào", "đâu"],
    "WHEN": ["khi nào", "lúc nào", "thời gian nào", "năm nào"],
    "WHY": ["tại sao", "vì sao", "do đâu"],
    "WHAT": ["là gì", "cái gì", "gì", "những gì"],
    "HOW": ["thế nào", "như thế nào", "bằng cách nào", "làm sao"],
    "OTHER": []
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
