# data_agumentor/data_augmentor.py
"""
Vietnamese Question Generation Data Augmentor
P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
           = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)
"""
import copy
import math
import numpy as np
from collections import Counter
import underthesea
from underthesea import ner, pos_tag, chunk, word_tokenize
from typing import List, Dict, Tuple
import pickle
import os

from data_agumentor.config import (
    NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE,
    FUNCTION_WORDS_LIST,
    QUESTION_TYPES,
    QUESTION_PATTERNS,
)


def val2bin(input_val: float, min_val: float, max_val: float, bin_width: float) -> int:
    """Convert value to bin number"""
    if min_val <= input_val <= max_val:
        return math.ceil((input_val - min_val) / bin_width)
    elif input_val > max_val:
        return math.ceil((max_val - min_val) / bin_width) + 1
    return -1


def get_token2char(tokens: List[str], text: str) -> Tuple[Dict, Dict]:
    """Create mappings between token indices and character positions"""
    token2idx = {}
    idx2token = {}
    current_pos = 0

    for i, token in enumerate(tokens):
        token_start = text.find(token, current_pos)
        if token_start == -1:
            continue

        token_end = token_start + len(token) - 1
        token2idx[i] = (token_start, token_end)

        for char_pos in range(token_start, token_end + 1):
            idx2token[char_pos] = i

        current_pos = token_end + 1

    return token2idx, idx2token


def str_find(text: str, token_list: List[str]) -> Tuple[int, int]:
    """Find position of token sequence in text"""
    search_text = " ".join(token_list)
    start_pos = text.find(search_text)

    if start_pos >= 0:
        return start_pos, start_pos + len(search_text) - 1
    return -1, -1


def get_chunks(sentence: str):
    """Get chunks from Vietnamese sentence"""
    # Get tokens and POS tags
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(sentence)

    # Get NER tags
    ner_tags = ner(sentence)
    ner_dict = {}

    print("NER tags:", ner_tags)

    # Handle NER tags - each tag is (word, pos, chunk, ner)
    current_entity = []
    current_type = None

    for tag_tuple in ner_tags:
        word, _, _, ner_tag = tag_tuple  # Unpack all 4 values

        if ner_tag.startswith('B-'):  # Beginning of entity
            if current_entity:
                ner_dict[' '.join(current_entity)] = current_type
                current_entity = []
            current_type = ner_tag[2:]  # Remove B- prefix
            current_entity.append(word)
        elif ner_tag.startswith('I-'):  # Inside of entity
            current_entity.append(word)
        else:  # Outside of entity (O tag)
            if current_entity:
                ner_dict[' '.join(current_entity)] = current_type
                current_entity = []
            current_type = None

    # Add last entity if exists
    if current_entity:
        ner_dict[' '.join(current_entity)] = current_type

    # Get chunks from underthesea
    chunk_results = underthesea.chunk(sentence)
    chunklist = []

    current_pos = 0
    for chunk_info in chunk_results:
        # underthesea chunk returns (text, type, tag)
        chunk_text = chunk_info[0]  # Get text from chunk tuple
        chunk_type = chunk_info[1]  # Get type from chunk tuple

        # Find chunk position
        chunk_start = -1
        for i in range(current_pos, len(tokens)):
            if chunk_text.startswith(tokens[i]):
                chunk_start = i
                break

        if chunk_start != -1:
            # Use word_tokenize for consistent tokenization
            chunk_tokens = word_tokenize(chunk_text)
            chunk_end = chunk_start + len(chunk_tokens) - 1
            current_pos = chunk_end + 1

            # Get NER tag from the original NER tags
            ner_tag = "UNK"
            for tag_tuple in ner_tags:
                if tag_tuple[0] == chunk_text:
                    if tag_tuple[3].startswith('B-'):
                        ner_tag = tag_tuple[3][2:]  # Remove B- prefix
                        break

            # If not found, try matching with ner_dict
            if ner_tag == "UNK":
                for ent_text, ent_type in ner_dict.items():
                    if chunk_text in ent_text or ent_text in chunk_text:
                        ner_tag = ent_type
                        break

            chunklist.append((
                ner_tag,
                chunk_type,
                chunk_tokens,
                chunk_start,
                chunk_end
            ))

    return chunklist, None, tokens


def get_distance(token1_idx: int, token2_idx: int, tokens: List[str]) -> int:
    """Calculate distance between two tokens"""
    return abs(token1_idx - token2_idx)


def select_answers(
    sentence: str,
    sample_probs: Dict,
    num_sample_answer: int = 5,
    answer_length_bin_width: int = 3,
    answer_length_min_val: int = 0,
    answer_length_max_val: int = 30,
    max_sample_times: int = 20,
) -> Tuple[List, List, None, List]:
    """Select potential answers from sentence"""
    # Get chunks
    chunklist, _, tokens = get_chunks(sentence)
    token2idx, idx2token = get_token2char(tokens, sentence)

    # Calculate answer probabilities
    chunk_ids = list(range(len(chunklist)))
    a_probs = []

    for chunk in chunklist:
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        a_tag = f"{chunk_pos_tag}-{chunk_ner_tag}"
        a_length = len(chunk[2])

        # Get length bin
        a_length_bin = val2bin(
            a_length,
            answer_length_min_val,
            answer_length_max_val,
            answer_length_bin_width,
        )

        # Get probability
        a_condition = f"{a_tag}_{a_length_bin}"
        if (
            a_condition in sample_probs["a"]
            and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE
        ):
            a_probs.append(sample_probs["a"][a_condition])
        else:
            a_probs.append(1)

    # Sample answers
    sampled_answers = []
    sampled_ids = set()

    for _ in range(max_sample_times):
        if len(sampled_answers) >= num_sample_answer:
            break

        # Sample chunk
        chunk_id = np.random.choice(
            chunk_ids, p=np.array(a_probs) / sum(a_probs))
        if chunk_id in sampled_ids:
            continue

        sampled_ids.add(chunk_id)
        chunk = chunklist[chunk_id]

        try:
            # Get character positions
            char_start, char_end = str_find(sentence, chunk[2])
            if char_start < 0:
                continue

            # Create BIO tags
            bio_ids = ["O"] * len(tokens)
            bio_ids[chunk[3]: chunk[4] + 1] = ["I"] * (chunk[4] - chunk[3] + 1)
            bio_ids[chunk[3]] = "B"

            sampled_answers.append(
                (
                    " ".join(chunk[2]),  # answer text
                    char_start,  # character start
                    char_end,  # character end
                    chunk[3],  # token start
                    chunk[4],  # token end
                    bio_ids,  # BIO tags
                    chunk[1],  # POS tag
                    chunk[0],  # NER tag
                )
            )

        except:
            continue

    return sampled_answers, chunklist, None, tokens


def select_clues(
    chunklist: List,
    tokens: List[str],
    sample_probs: Dict,
    selected_answer: Tuple,
    num_sample_clue: int = 2,
    clue_dep_dist_bin_width: int = 2,
    clue_dep_dist_min_val: int = 0,
    clue_dep_dist_max_val: int = 20,
    max_sample_times: int = 20,
) -> List[Dict]:
    """Select clue chunks for Vietnamese text"""
    (
        answer_text,
        char_st,
        char_ed,
        ans_start,
        ans_end,
        answer_bio_ids,
        answer_pos_tag,
        answer_ner_tag,
    ) = selected_answer

    # Calculate clue probabilities
    c_probs = []
    valid_chunks = []

    for chunk in chunklist:
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        c_tag = f"{chunk_pos_tag}-{chunk_ner_tag}"

        # Skip if chunk overlaps with answer
        if (chunk[3] >= ans_start and chunk[3] <= ans_end) or (
            chunk[4] >= ans_start and chunk[4] <= ans_end
        ):
            continue

        # Calculate distance to answer
        dist = min(abs(chunk[3] - ans_end), abs(chunk[4] - ans_start))

        # Get distance bin
        dist_bin = val2bin(
            dist, clue_dep_dist_min_val, clue_dep_dist_max_val, clue_dep_dist_bin_width
        )

        # Get probability
        c_condition = f"{c_tag}_{dist_bin}"
        if (
            c_condition in sample_probs["c|a"]
            and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE
        ):
            c_probs.append(sample_probs["c|a"][c_condition])
            valid_chunks.append(chunk)

    if not valid_chunks:
        return []

    # Sample clues
    sampled_clues = []
    sampled_ids = set()

    for _ in range(max_sample_times):
        if len(sampled_clues) >= num_sample_clue:
            break

        # Sample chunk
        chunk_id = np.random.choice(
            range(len(valid_chunks)), p=np.array(c_probs) / sum(c_probs)
        )

        if chunk_id in sampled_ids:
            continue

        sampled_ids.add(chunk_id)
        chunk = valid_chunks[chunk_id]

        # Create clue info
        clue_text = " ".join(chunk[2])
        clue_binary_ids = [0] * len(tokens)
        clue_binary_ids[chunk[3]: chunk[4] + 1] = [1] * \
            (chunk[4] - chunk[3] + 1)

        sampled_clues.append(
            {
                "clue_text": clue_text,
                "clue_binary_ids": clue_binary_ids,
                "clue_tag": chunk[1],
                "distance": min(abs(chunk[3] - ans_end), abs(chunk[4] - ans_start)),
            }
        )

    return sampled_clues


def select_question_types(
    sample_probs: Dict,
    selected_answer: Tuple,
    num_sample_style: int = 2,
    max_sample_times: int = 20,
) -> List[str]:
    """Select appropriate question types for Vietnamese answer"""
    (answer_text, _, _, _, _, _, answer_pos_tag, answer_ner_tag) = selected_answer
    a_tag = f"{answer_pos_tag}-{answer_ner_tag}"

    # Calculate style probabilities
    s_probs = []
    for style in QUESTION_TYPES:
        s_condition = f"{style}_{a_tag}"
        if s_condition in sample_probs["s|c,a"]:
            s_probs.append(sample_probs["s|c,a"][s_condition])
        else:
            # Default probabilities based on answer properties
            if answer_ner_tag == "PER" and style == "WHO":
                s_probs.append(2.0)
            elif answer_ner_tag == "LOC" and style == "WHERE":
                s_probs.append(2.0)
            elif answer_ner_tag == "TIME" and style == "WHEN":
                s_probs.append(2.0)
            elif answer_ner_tag == "NUM" and style == "HOW":
                s_probs.append(2.0)
            else:
                s_probs.append(1.0)

    # Normalize probabilities
    s_probs = np.array(s_probs) / sum(s_probs)

    # Sample styles
    sampled_styles = []
    for _ in range(max_sample_times):
        if len(sampled_styles) >= num_sample_style:
            break

        style = np.random.choice(QUESTION_TYPES, p=s_probs)
        if style not in sampled_styles:
            sampled_styles.append(style)

    return sampled_styles


def get_dataset_info(examples: List[Dict], sent_limit: int = 100) -> List[Dict]:
    """Extract dataset information from Vietnamese QA examples"""
    examples_with_info = []

    for e in examples:
        try:
            sentence = e["ans_sent"]
            question = e["question"]
            answer = e["answer_text"]
            answer_start = e["answer_start"]

            # Get chunks
            chunklist, _, tokens = get_chunks(sentence)

            # Get answer info
            answer_pos_tag = "UNK"
            answer_ner_tag = "UNK"

            for chunk in chunklist:
                if answer == " ".join(chunk[2]):
                    answer_ner_tag = chunk[0]
                    answer_pos_tag = chunk[1]
                    break

            # Get question type
            for qtype, patterns in QUESTION_PATTERNS.items():
                question_lower = question.lower()
                if any(p in question_lower for p in patterns):
                    question_type = qtype
                    break
            else:
                question_type = "OTHER"

            examples_with_info.append(
                {
                    "sentence": sentence,
                    "question": question,
                    "answer_text": answer,
                    "answer_start": answer_start,
                    "answer_tag": f"{answer_pos_tag}-{answer_ner_tag}",
                    "answer_length": len(answer.split()),
                    "question_type": question_type,
                }
            )

        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue

    return examples_with_info


def get_sample_probs(
    examples: List[Dict], answer_length_bins: int = 10, clue_distance_bins: int = 10
) -> Dict:
    """Calculate sampling probabilities from Vietnamese dataset"""
    examples_with_info = get_dataset_info(examples)

    # Initialize counters
    answer_probs = Counter()  # P(a|tag,length)
    clue_probs = Counter()  # P(c|tag,distance)
    style_probs = Counter()  # P(s|answer_tag)

    for e in examples_with_info:
        # Answer probabilities
        a_tag = e["answer_tag"]
        a_len_bin = val2bin(e["answer_length"], 0, 30, 30 / answer_length_bins)
        answer_probs[f"{a_tag}_{a_len_bin}"] += 1

        # Question type probabilities
        style_probs[f"{e['question_type']}_{a_tag}"] += 1

    return {"a": answer_probs, "c|a": clue_probs, "s|c,a": style_probs}


def augment_qg_data(
    sentence: str,
    sample_probs: Dict,
    num_sample_answer: int = 5,
    num_sample_clue: int = 2,
    num_sample_style: int = 2,
    answer_length_bin_width: int = 3,
    answer_length_min_val: int = 0,
    answer_length_max_val: int = 30,
    clue_dep_dist_bin_width: int = 2,
    clue_dep_dist_min_val: int = 0,
    clue_dep_dist_max_val: int = 20,
    max_sample_times: int = 20,
) -> Dict:
    """Generate augmented QA data for Vietnamese sentence"""
    sampled_infos = []

    # Sample answers
    sampled_answers, chunklist, _, tokens = select_answers(
        sentence,
        sample_probs,
        num_sample_answer,
        answer_length_bin_width,
        answer_length_min_val,
        answer_length_max_val,
        max_sample_times,
    )

    # For each answer, sample clues and styles
    for ans in sampled_answers:
        answer_info = {
            "answer_text": ans[0],
            "char_start": ans[1],
            "char_end": ans[2],
            "answer_bio_ids": ans[5],
            "answer_chunk_tag": ans[6],
        }

        # Sample question styles
        styles = select_question_types(
            sample_probs, ans, num_sample_style, max_sample_times
        )

        # Sample clues
        clues = select_clues(
            chunklist,
            tokens,
            sample_probs,
            ans,
            num_sample_clue,
            clue_dep_dist_bin_width,
            clue_dep_dist_min_val,
            clue_dep_dist_max_val,
            max_sample_times,
        )

        sampled_infos.append(
            {"answer": answer_info, "styles": styles, "clues": clues})

    return {"context": sentence, "samples": sampled_infos}


if __name__ == "__main__":
    print("=== Testing Vietnamese Question Generation Data Augmentation ===\n")

    # 1. Test with sample training data
    print("1. Testing with sample training data:")
    training_examples = [
        {
            "ans_sent": "Nguyễn Du là một nhà thơ lớn của Việt Nam",
            "question": "Ai là nhà thơ lớn của Việt Nam?",
            "answer_text": "Nguyễn Du",
            "answer_start": 0
        },
        {
            "ans_sent": "Hà Nội là thủ đô của nước Việt Nam từ năm 1010",
            "question": "Đâu là thủ đô của Việt Nam?",
            "answer_text": "Hà Nội",
            "answer_start": 0
        },
        {
            "ans_sent": "Trường Đại học Bách Khoa Hà Nội được thành lập vào năm 1956",
            "question": "Khi nào Trường Đại học Bách Khoa Hà Nội được thành lập?",
            "answer_text": "năm 1956",
            "answer_start": 55
        }
    ]

    print("\nCalculating sampling probabilities...")
    try:
        sample_probs = get_sample_probs(training_examples)
        print("Sample probabilities calculated")
        print("Probabilities:", sample_probs)
    except Exception as e:
        print(f"Error calculating probabilities: {str(e)}")
        sample_probs = {
            "a": Counter(),
            "c|a": Counter(),
            "s|c,a": Counter()
        }

    # 2. Test data augmentation
    print("\n2. Testing data augmentation:")
    test_sentences = [
        "Nguyễn Du là một nhà thơ lớn của Việt Nam",
        "Hà Nội là thủ đô của nước Việt Nam từ năm 1010",
        "Trường Đại học Bách Khoa Hà Nội được thành lập vào năm 1956",
        "Việt Nam có diện tích 331.212 km vuông",
        "Sông Hồng là con sông lớn nhất ở miền Bắc Việt Nam"
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest sentence {i}: {sentence}")

        try:
            # Get chunks for debugging
            chunks, _, tokens = get_chunks(sentence)
            print(f"Chunks found: {len(chunks)}")
            for chunk in chunks:
                print(f"- {' '.join(chunk[2])} ({chunk[1]}, {chunk[0]})")

            result = augment_qg_data(
                sentence=sentence,
                sample_probs=sample_probs,
                num_sample_answer=3,
                num_sample_clue=2,
                num_sample_style=2
            )

            print(f"\nGenerated {len(result['samples'])} samples:")
            for j, sample in enumerate(result['samples'], 1):
                print(f"\nSample {j}:")
                print("Answer:", sample['answer']['answer_text'])
                print("Question styles:", sample['styles'])
                if sample['clues']:
                    print("Clues:", [c['clue_text'] for c in sample['clues']])

        except Exception as e:
            print(f"Error processing sentence: {str(e)}")
            import traceback
            traceback.print_exc()
