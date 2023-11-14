from datetime import datetime
import json
# from utils.utils import normalize_text
from tqdm import tqdm
import random
import underthesea
from underthesea import sent_tokenize, word_tokenize
# from constants import INFO_QUESTION_TYPES, INFO_QUESTION_TYPES_MAPPING, Q_TYPE2ID_DICT

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
  "Where": ["ở đâu", "nơi nào", "đâu"],
  "When": ["khi nào", "lúc nào"],
  "Why": ["tại sao", "vì sao", "do đâu"], 
  "How": ["thế nào", "bao nhiêu"],
  "What": ["cái gì", "là gì", "gì"], 
}

FUNCTION_TOKENS_FILE_PATH = 'Datasets/function-tokens.txt'
f_func_tokens = open(FUNCTION_TOKENS_FILE_PATH, "r", encoding="utf-8")
func_tokens = f_func_tokens.readlines()
FUNCTION_TOKEN_LIST = [word.rstrip() for word in func_tokens]

def get_raw_examples(filename, level='paragraph' ,debug=False, debug_length=20):
  print(f"Start get {filename} raw examples ...")
  start = datetime.now()
  file = open(filename)
  data = json.load(file)
  articles = data['data']
  num_examples = 0
  raw_examples = []
  for article in tqdm(articles):
    paragraphs = article['paragraphs']
    for paragraph in paragraphs:
      paragraph_content = paragraph['context']
      qa_pairs = paragraph['qas']
      for qa_pair in qa_pairs:
        question = qa_pair['question']
        answers = qa_pair['answers']
        for answer in answers:  
          example = {
            'paragraph': paragraph_content,
            'question': question,
            'answer_text': answer['text'],
            'answer_start': answer['answer_start']
          }
          if level == 'sentence': 
            example = build_sentence_level_for(example)
          raw_examples.append(example)
          num_examples += 1
          if debug and num_examples >= debug_length:
            break
  print(("Time of get raw examples: {}").format(datetime.now() - start))
  print("Number of raw examples: ", len(raw_examples))
  return raw_examples

def normalize_text(text):
    """
    Replace some special characters in text.
    """
    # NOTICE: don't change the text length.
    # Otherwise, the answer position is changed.
    text = text.replace("''", '" ').replace("``", '" ')
    return text

def build_sentence_level_for(example):
  sentences = sent_tokenize(example['paragraph'])
  tokens_passed = 0
  # import pdb; pdb.set_trace()
  for sentence in sentences:
    if example['answer_start'] < tokens_passed + len(sentence): # current sentence contains answer
      # assert(example['answer_text'] in sentence)
      if example['answer_text'] in sentence:
      # replace examples's paragraph by the current sentence 
        example['paragrah'] = sentence
      # re-calculate the answer_start
        new_start_token = example['answer_start'] - tokens_passed
        example['answer_start'] = new_start_token
        break
    else:
      tokens_passed += len(sentence) + 1 # one more for space
  return example
    
def get_question_type(question):
  """
  Given a string question, return its type name and type id.
  :param question: question string.
  :return: (question_type_str, question_type_id)
  """
  words = question.split()
  for word in words:
    for i in INFO_QUESTION_TYPES:
      for j in INFO_QUESTION_TYPES_MAPPING[i]:
        if j.upper() in word.upper():
          return (i, Q_TYPE2ID_DICT[i])
  return ("Other", Q_TYPE2ID_DICT["Other"])

def get_chunks(sentence): #chunking
    """
    Input a sentence, output a list of its chunks (ner_tag, pos_tag, leaves_without_position, st, ed).
    Such as ('PERSON', 'NP', ['Beyoncé', 'Giselle', 'Knowles-Carter'], 0, 2).
    """
    #each chunk format: <word, pos, pos>
    original_chunks = underthesea.chunk(sentence) 
    chunk_list = []
      
    for chunk in original_chunks:
      chunk_text, _, _ = chunk
      chunk_tokens = chunk_text.split()
      chunk_list.append((chunk_text, chunk_tokens))
      
    return chunk_list 

def get_clue_info(question, sentence, answer):
    example = {
        "question": question,
        "ans_sent": sentence,
        "answer_text": answer}

    # doc = underthesea.ner(sentence)
    chunklist = get_chunks(sentence)
        
    # example["ans_sent_tokens"] = word_tokenize(sentence)
    example["ques_tokens"] = word_tokenize(question)
    # example["ans_sent_doc"] = doc

    clue_rank_scores = []
    for chunk in chunklist:
        candidate_clue_text, candidate_clue_tokens = chunk
      
        ques_lower = " ".join(example["ques_tokens"]).lower()
        
        ques_tokens = [t.lower() for t in example["ques_tokens"]]
        candidate_clue_is_content = [int(w.lower() not in FUNCTION_TOKEN_LIST) for w in candidate_clue_tokens]

        candidate_clue_tokens_in_ques = [candidate_clue_tokens[i] for i in range(len(candidate_clue_tokens)) if candidate_clue_tokens[i].lower() in ques_tokens]
        candidate_clue_content_tokens = [candidate_clue_tokens[i] for i in range(len(candidate_clue_tokens)) if candidate_clue_is_content[i] == 1]
        candidate_clue_content_tokens_in_ques = [candidate_clue_content_tokens[i] for i in range(len(candidate_clue_content_tokens)) if candidate_clue_content_tokens[i].lower() in ques_tokens]

        score = 0
        if  len(candidate_clue_tokens_in_ques) == len(candidate_clue_tokens) and \
                sum(candidate_clue_is_content) > 0 and \
                candidate_clue_tokens[0].lower() not in FUNCTION_TOKEN_LIST:
            score += len(candidate_clue_content_tokens_in_ques) # number of overlap token between q and c
            score += int(candidate_clue_text.lower() in ques_lower) # binary core if q contain chunk c
        clue_rank_scores.append(score)
    max_score = max(clue_rank_scores)
    
    clue_info = {
        "clue_text": None,
        "clue_tokens": None,
        "clue_length": 0,
    }
    
    if len(clue_rank_scores) != 0 and max_score > 0:
        selected_chunk_text, selected_chunk_tokens = chunklist[clue_rank_scores.index(max_score)]
        clue_info['clue_text'] = selected_chunk_text
        clue_info['clue_tokens'] = selected_chunk_tokens
        clue_info['clue_length'] = len(selected_chunk_text)

    return clue_info
  
def get_processed_examples(raw_examples, debug=False, debug_length=20, shuffle=True):
  print("Start transform raw examples to processed examples...")
  start = datetime.now()
  examples = []
  meta = {}
  meta["num_q"] = 0
  # num_spans_len_error = 0
  # num_not_match_error = 0
  for example in tqdm(raw_examples):
    # paragraph info (here is sentence) 
    paragraph = normalize_text(example["paragraph"]) #replace special char
    # ans_sent_doc = word_tokenize(paragraph) # tokenize the sentence
    
    #question
    ques = normalize_text(example["question"])
    # ques = "<sos> " + ques + " <eos>"  # notice: this is important for QG
    ques_type, _ = get_question_type(example["question"]) # style
    
    # answer info
    answer_text = normalize_text(example["answer_text"])
    answer_start = example["answer_start"]
    
    clue_info = get_clue_info(question=ques, sentence=paragraph, answer=answer_text)
    if clue_info["clue_text"] is not None:
      clue_text = clue_info["clue_text"]
      clue_start = paragraph.find(clue_text)
    example = {
            "paragraph": paragraph,

            "question": ques,
            "ques_type": ques_type,  # string type

            "answer": answer_text,
            "answer_start": answer_start,

            "clue": clue_text,
            "clue_start": clue_start,
            "para_id": meta["num_q"]}
    examples.append(example)
    meta["num_q"] += 1
    
    if debug and meta["num_q"] >= debug_length:
      break

    if shuffle:
      random.shuffle(examples)

  # print("num_not_match_error: ", num_not_match_error)
  # print("num_spans_len_error: ", num_spans_len_error)
  print(("Time of get processed examples: {}").format(datetime.now() - start))
  print("Number of processed examples: ", len(examples))
  return examples
  
if __name__ == "__main__":
  raws = get_raw_examples('Datasets/ViQuAD1.0/dev_ViQuAD.json', level='sentence', debug=True, debug_length=1)
  processed = get_processed_examples(raws, debug=True)
  with open(r'debug.txt', 'w') as fp:
    for e in processed:
      fp.write("%s\n" % e)
    print('Done')