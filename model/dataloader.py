import os
import torch
import numpy as np
from tqdm import tqdm
from data_loader.dataloader import get_raw_examples, get_processed_examples


def get_dataset(tokenizer, dataset_cache, path, debug=False, debug_length=20):
    # Load question data
    if dataset_cache and os.path.isfile(dataset_cache):  #!!! NOTICE: make sure dataset_cache is correct version.
        print("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        return data

    data = get_positional_dataset_from_file(tokenizer, file=path, debug=debug, debug_length=debug_length)

    if dataset_cache:
        torch.save(data, dataset_cache)

    print("Dataset cached at %s", dataset_cache)

    return data
  
def get_position(para_ids, ans_ids, ans_prefix_ids):
    diff_index = -1
    # Find the first token where the paragraph and answer prefix differ
    for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
        if pid != apid:
            diff_index = i
            break
    if diff_index == -1:
        diff_index = min(len(ans_prefix_ids), len(para_ids))
    # Starting from this token, we take a conservative overlap
    return (diff_index, min(diff_index + len(ans_ids), len(para_ids)))
  
  
def get_positional_dataset_from_file(tokenizer, file, debug=False, debug_length=20):
    data = get_raw_examples(file, debug, debug_length)
    data = get_processed_examples(data, debug)
    truncated_sequences = 0
    for inst in tqdm(data):
        inst['answer_position'] = inst['answer_start']
        clue_exist = (inst['clue_start'] is not None)
        if clue_exist:
            inst['clue_position'] = inst['clue_start']

        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position']])

        if clue_exist:
            tokenized_clue = tokenizer.tokenize(inst['clue'])
            tokenized_clue_prefix = tokenizer.tokenize(inst['paragraph'][:inst['clue_position']])
        else:
            tokenized_clue = []
        # if question type is Other, using <mask> token to represent the question stype
        if inst['ques_type'] == "Other":
          tokenized_qtype = '<mask>'
        else:
          tokenized_qtype = tokenizer.tokenize(inst['ques_type_text'])

        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + len(tokenized_clue) + len(tokenized_qtype) + 6

        if total_seq_len > tokenizer.model_max_length:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.model_max_length + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + len(tokenized_clue) + len(tokenized_qtype) + 6 < tokenizer.model_max_length

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
        inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)

        if clue_exist:
            inst['clue'] = tokenizer.convert_tokens_to_ids(tokenized_clue)
            clue_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_clue_prefix)
            inst['clue_position_tokenized'] = get_position(inst['paragraph'], inst['clue'], clue_prefix_ids)

        inst['style'] = tokenizer.convert_tokens_to_ids(tokenized_qtype)
        pass

    print("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data

