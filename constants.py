# question type
QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "What", "How", "Other"]
INFO_QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "What", "How"]
Q_TYPE2ID_DICT = {
    "What": 0, "Who": 1, "How": 2,
    "Where": 3, "When": 4, "Why": 5, "Other": 7}
INFO_QUESTION_TYPES_MAPPING = {
  "Who": ["ai", "người nào"],
  "Where": ["Ở đâu", "đâu", "nơi nào"],
  "When": ["khi nào", "lúc nào"],
  "Why": ["tại sao", "vì sao", "do đâu"],
  "What": ["cái gì", "là gì"], 
  "How": ["như thế nào", "bao nhiêu"],
}