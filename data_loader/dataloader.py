from datetime import datetime
import json
from utils.utils import normalize_text
from tqdm import tqdm
from underthesea import sent_tokenize, word_tokenize
from constants import INFO_QUESTION_TYPES, INFO_QUESTION_TYPES_MAPPING, Q_TYPE2ID_DICT

def get_raw_example(filename, level='paragraph' ,debug=False, debug_length=20):
  print(f"Start get {filename} raw examples ...")
  start = datetime.now()
  file = open(f"{filename}.json")
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
      tokens_passed += len(sentence) + 1 # one more for spca
  return example
  
def get_processed_example(raw_examples, debug=False, debug_length=20, shuffle=True):
  print("Start transform raw examples to processed examples...")
  start = datetime.now()
  examples = []
  meta = {}
  meta["num_q"] = 0
  num_spans_len_error = 0
  num_not_match_error = 0
  for example in tqdm(raw_examples):
    # paragraph info (here is sentence) 
    paragraph = normalize_text(example["paragraph"]) #replace special char
    ans_sent_doc = word_tokenize(paragraph) # tokenize the sentence
    ans_sent_tokens = [token.text for token in ans_sent_doc] # array of token'texts
    
    #question
    ques = normalize_text(e["question"])
    # ques = "<sos> " + ques + " <eos>"  # notice: this is important for QG
    ques_doc = NLP(ques)
    ques_type, _ = get_question_type(e["question"]) # style
    
    # answer info
    answer_text = normalize_text(e["answer_text"])
    answer_start = e["answer_start"]
    answer_end = answer_start + len(answer_text)
    answer_span = []
 
 
    
def get_question_type(question):
  """
  Given a string question, return its type name and type id.
  :param question: question string.
  :return: (question_type_str, question_type_id)
  """
  words = question.split()
  for word in words:
    for i in range(len(INFO_QUESTION_TYPES)):
      for j in INFO_QUESTION_TYPES[i]:
        if INFO_QUESTION_TYPES_MAPPING[j].upper() in word.upper():
          return (INFO_QUESTION_TYPES[i], Q_TYPE2ID_DICT[INFO_QUESTION_TYPES[i]])
  return ("Other", Q_TYPE2ID_DICT["Other"])
  
if __name__ == "__main__":
  examples = get_raw_example('Datasets/ViQuAD1.0/dev_ViQuAD', debug=True, debug_length=1)
  with open(r'debug.txt', 'w') as fp:
    for e in examples:
      fp.write("%s\n" % e)
    print('Done')