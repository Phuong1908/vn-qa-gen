# data_agumentor/data_augmentor.py
"""
Vietnamese Question Generation Data Augmentor
P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
           = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)
"""
from data_agumentor.config import (
    NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE,
    QUESTION_TYPES,
    QUESTION_PATTERNS,
)
from tqdm import tqdm
import math
import numpy as np
from collections import Counter
import underthesea
from underthesea import ner, pos_tag, chunk, word_tokenize
from typing import List, Dict, Tuple
from .vncorenlp_wrapper import VnCorenlpHandler
import os
import pickle

vncore_handler = VnCorenlpHandler(
    "/Users/phuongnguyen/study/vn-qa-gen/vncorenlp")


def load_sample_probs(self):
    """Load sampling probabilities from ViQuAD"""
    viquad_probs_file = "viquad_sample_probs.pkl"

    if os.path.exists(viquad_probs_file):
        print(f"Loading probabilities from {viquad_probs_file}")
        with open(viquad_probs_file, 'rb') as f:
            probs = pickle.load(f)
            print(f"Loaded probabilities with:")
            print(f"- {len(probs['a'])} answer patterns")
            print(f"- {len(probs['c|a'])} clue patterns")
            print(f"- {len(probs['s|c,a'])} style patterns")
            return probs
    else:
        raise FileNotFoundError(
            f"Sample probabilities file not found at {viquad_probs_file}")


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
    """Get chunks from Vietnamese sentence with improved compound noun handling"""
    # Initial tokenization and tagging
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(sentence)
    ner_tags = ner(sentence)
    chunk_results = underthesea.chunk(sentence)

    # Helper function to check if tokens form a valid compound
    def is_valid_compound(start_idx: int, end_idx: int,
                          pos_tags: List[tuple], ner_tags: List[tuple]) -> bool:
        """Check if tokens from start to end form a valid compound based on linguistic patterns"""
        if start_idx >= end_idx:
            return False

        # Get POS and NER tags for the span
        span_pos = [tag[1] for tag in pos_tags[start_idx:end_idx + 1]]
        span_ner = [tag[3] for tag in ner_tags[start_idx:end_idx + 1]]

        # Check patterns that indicate compounds:

        # 1. Continuous proper nouns or nouns
        if all(pos in ['N', 'Np', 'Nc'] for pos in span_pos):
            return True

        # 2. Named entity continuation
        if any(tag.startswith('B-') for tag in span_ner) and \
           any(tag.startswith('I-') for tag in span_ner):
            return True

        # 3. Noun + Modifier pattern (e.g., "sông lớn", "thành phố cổ")
        if len(span_pos) == 2 and span_pos[0] in ['N', 'Np'] and span_pos[1] in ['A', 'N']:
            return True

        # 4. Location compounds (e.g., "miền Bắc", "thành phố Hà Nội")
        if span_pos[0] in ['N', 'Np'] and any(tag.startswith('B-LOC') or tag.startswith('I-LOC') for tag in span_ner):
            return True

        return False

    # First pass: Process NER tags to identify base entities
    compound_dict = {}
    current_entity = []
    current_start = None
    current_type = None

    for i, (word, pos, _, ner_tag) in enumerate(ner_tags):
        if ner_tag.startswith('B-'):
            if current_entity:
                compound_text = " ".join(current_entity)
                compound_dict[compound_text] = {
                    'type': current_type,
                    'start': current_start,
                    'end': i - 1,
                    'pos': 'Np'  # Mark as proper noun
                }
            current_entity = [word]
            current_type = ner_tag[2:]
            current_start = i
        elif ner_tag.startswith('I-'):
            current_entity.append(word)
        else:
            if current_entity:
                compound_text = " ".join(current_entity)
                compound_dict[compound_text] = {
                    'type': current_type,
                    'start': current_start,
                    'end': i - 1,
                    'pos': 'Np'
                }
                current_entity = []
            # Check for potential compound start
            if pos in ['N', 'Np']:
                current_entity = [word]
                current_start = i
                current_type = 'UNK'

    if current_entity:
        compound_text = " ".join(current_entity)
        compound_dict[compound_text] = {
            'type': current_type,
            'start': current_start,
            'end': len(ner_tags) - 1,
            'pos': 'Np'
        }

    # Second pass: Look for additional compounds using linguistic patterns
    i = 0
    while i < len(tokens) - 1:
        # Try extending current token to form compounds
        for j in range(i + 1, min(i + 5, len(tokens))):  # Look up to 4 tokens ahead
            if is_valid_compound(i, j, pos_tags, ner_tags):
                compound_text = " ".join(tokens[i:j + 1])
                # Check if this compound or its parts aren't already in compound_dict
                if not any(existing_comp for existing_comp in compound_dict
                           if compound_text in existing_comp or existing_comp in compound_text):
                    # Determine compound type from NER tags
                    compound_ner = [tag[3] for tag in ner_tags[i:j + 1]]
                    compound_type = 'UNK'
                    for tag in compound_ner:
                        if tag.startswith('B-'):
                            compound_type = tag[2:]
                            break

                    compound_dict[compound_text] = {
                        'type': compound_type,
                        'start': i,
                        'end': j,
                        'pos': 'Np' if any(tag[1] == 'Np' for tag in pos_tags[i:j + 1]) else 'N'
                    }
        i += 1

    # Build final chunklist
    chunklist = []
    processed_positions = set()

    # Add compounds first
    for compound_text, info in sorted(compound_dict.items(), key=lambda x: x[1]['start']):
        chunk_tokens = word_tokenize(compound_text)
        for i in range(info['start'], info['end'] + 1):
            processed_positions.add(i)

        chunklist.append((
            info['type'],
            info['pos'],
            chunk_tokens,
            info['start'],
            info['end']
        ))

    # Add remaining single tokens
    for i, (token, pos, _, ner_tag) in enumerate(ner_tags):
        if i not in processed_positions and pos not in ['CH', 'C', 'E', 'R', 'T']:
            ner_type = ner_tag[2:] if ner_tag.startswith('B-') else 'UNK'
            chunklist.append((
                ner_type,
                pos,
                [token],
                i,
                i
            ))

    return chunklist, None, tokens


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
    print(f"\nFound chunks: {len(chunklist)}")

    # Filter valid chunks
    valid_chunks = []
    valid_probs = []

    for chunk in chunklist:
        chunk_text = " ".join(chunk[2])
        print(f"Evaluating chunk: {chunk_text} ({chunk[1]}-{chunk[0]})")

        if any([
            chunk_text.lower() in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE,
            chunk[1] in ['CH', 'C', 'E', 'R', 'T'],
        ]):
            continue

        # Calculate base score based on chunk type
        base_score = 1.0

        # # Prefer named entities
        # if chunk[0] != "UNK":
        #     base_score *= 3.0
        #     print(f"Named entity bonus for: {chunk_text}")

        # # Prefer nouns and noun phrases
        # if chunk[1] in ['N', 'Np']:
        #     base_score *= 2.0
        #     print(f"Noun bonus for: {chunk_text}")

        # # Prefer longer meaningful phrases
        # if len(chunk[2]) >= 2:
        #     base_score *= 1.5
        #     print(f"Length bonus for: {chunk_text}")

        # Get probability from learned distribution
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        a_tag = f"{chunk_pos_tag}-{chunk_ner_tag}"
        a_length = len(chunk[2])
        a_length_bin = val2bin(a_length, answer_length_min_val,
                               answer_length_max_val, answer_length_bin_width)

        a_condition = f"{a_tag}_{a_length_bin}"
        if a_condition in sample_probs["a"]:
            prob = sample_probs["a"][a_condition] * base_score
        else:
            prob = base_score

        print(f"Adding valid chunk: {chunk_text} with score {prob}")
        valid_chunks.append(chunk)
        valid_probs.append(prob)

    if not valid_chunks:
        print("No valid chunks found!")
        return [], chunklist, None, tokens

    # Normalize probabilities
    valid_probs = np.array(valid_probs)
    valid_probs = valid_probs / valid_probs.sum()

    # Sample answers
    sampled_answers = []
    sampled_ids = set()

    for _ in range(max_sample_times):
        if len(sampled_answers) >= num_sample_answer:
            break

        chunk_id = np.random.choice(len(valid_chunks), p=valid_probs)
        if chunk_id in sampled_ids:
            continue

        chunk = valid_chunks[chunk_id]
        try:
            # Get character positions
            char_start, char_end = str_find(sentence, chunk[2])
            if char_start < 0:
                print(
                    f"Could not find position for chunk: {' '.join(chunk[2])}")
                continue

            # Create BIO tags
            bio_ids = ["O"] * len(tokens)
            bio_ids[chunk[3]: chunk[4] + 1] = ["I"] * (chunk[4] - chunk[3] + 1)
            bio_ids[chunk[3]] = "B"

            sampled_answers.append(
                (
                    " ".join(chunk[2]),
                    char_start,
                    char_end,
                    chunk[3],
                    chunk[4],
                    bio_ids,
                    chunk[1],
                    chunk[0],
                )
            )
            sampled_ids.add(chunk_id)
            print(f"Added answer: {' '.join(chunk[2])}")

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue

    print(f"Selected {len(sampled_answers)} answers")
    return sampled_answers, chunklist, None, tokens


def select_clues(
    chunklist: List,
    tokens: List[str],
    sample_probs: Dict,
    selected_answer: Tuple,
    vncore_handler: VnCorenlpHandler,
    num_sample_clue: int = 2,
    clue_dep_dist_bin_width: int = 2,
    clue_dep_dist_min_val: int = 0,
    clue_dep_dist_max_val: int = 20,
    max_sample_times: int = 20,
) -> List[Dict]:
    """Select clue chunks with proper dependency distance"""
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

    # Get sentence text
    sentence = " ".join(tokens)

    # Calculate clue probabilities
    c_probs = []
    valid_chunks = []

    for chunk in chunklist:
        chunk_pos_tag = chunk[1]
        chunk_ner_tag = chunk[0]
        c_tag = f"{chunk_pos_tag}-{chunk_ner_tag}"

        if any([
            # Skip if chunk overlaps with answer
            (chunk[3] >= ans_start and chunk[3] <= ans_end) or
            (chunk[4] >= ans_start and chunk[4] <= ans_end),
            chunk[2][0].lower() in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE,
            chunk_pos_tag in ['CH', 'C', 'E', 'R', 'T'],
        ]):
            continue

        # Skip if chunk overlaps with answer
        if (chunk[3] >= ans_start and chunk[3] <= ans_end) or (
            chunk[4] >= ans_start and chunk[4] <= ans_end
        ):
            continue

        # Calculate dependency distance using VnCoreNLP with mapping
        distance_result = vncore_handler.get_dependency_distance(
            sentence,
            chunk[3],  # clue start
            ans_start,  # answer start
            tokens  # Pass underthesea tokens for mapping
        )

        if distance_result is None:
            continue

        dep_dist, path_desc = distance_result

        # Get distance bin
        dist_bin = val2bin(
            dep_dist,
            clue_dep_dist_min_val,
            clue_dep_dist_max_val,
            clue_dep_dist_bin_width
        )

        # Get probability
        c_condition = f"{c_tag}_{dist_bin}"
        if (
            c_condition in sample_probs["c|a"]
            and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE
        ):
            c_probs.append(sample_probs["c|a"][c_condition])
            # Store dependency info with chunk
            valid_chunks.append((chunk, dep_dist, path_desc))

    if not valid_chunks:
        return []

    # Sample clues using the calculated probabilities
    sampled_clues = []
    sampled_ids = set()

    for _ in range(max_sample_times):
        if len(sampled_clues) >= num_sample_clue:
            break

        # Sample chunk
        chunk_id = np.random.choice(
            range(len(valid_chunks)),
            p=np.array(c_probs) / sum(c_probs)
        )

        if chunk_id in sampled_ids:
            continue

        sampled_ids.add(chunk_id)
        chunk, dep_dist, path_desc = valid_chunks[chunk_id]

        # Create clue info
        clue_text = " ".join(chunk[2])
        clue_binary_ids = [0] * len(tokens)
        clue_binary_ids[chunk[3]: chunk[4] + 1] = [1] * \
            (chunk[4] - chunk[3] + 1)

        sampled_clues.append({
            "clue_text": clue_text,
            "clue_binary_ids": clue_binary_ids,
            "clue_tag": chunk[1],
            "distance": dep_dist,
            "dependency_path": path_desc  # Include dependency path info
        })

    return sampled_clues


def select_question_types(
    sample_probs: Dict,
    selected_answer: Tuple,
    num_sample_style: int = 2,
    max_sample_times: int = 20,
) -> List[str]:
    """Select question types based on P(s|a_tag) from training data"""
    answer_text, _, _, _, _, _, answer_pos_tag, answer_ner_tag = selected_answer
    a_tag = f"{answer_pos_tag}-{answer_ner_tag}"

    # Get style probabilities for this answer tag
    style_probs = []
    for style in QUESTION_TYPES:
        s_condition = f"{style}_{a_tag}"
        if s_condition in sample_probs["s|c,a"]:
            # Use learned probability
            style_probs.append(sample_probs["s|c,a"][s_condition])
        else:
            # Fallback probabilities based on answer properties
            if answer_ner_tag == "PER" and style == "WHO":
                style_probs.append(0.6)
            elif answer_ner_tag == "LOC" and style == "WHERE":
                style_probs.append(0.6)
            elif answer_ner_tag == "TIME" and style == "WHEN":
                style_probs.append(0.6)
            elif answer_ner_tag == "NUM" and style == "HOW":
                style_probs.append(0.6)
            else:
                style_probs.append(0.1)  # Small probability for other styles

    # Normalize probabilities
    style_probs = np.array(style_probs)
    style_probs = style_probs / style_probs.sum()

    # Sample styles
    sampled_styles = []
    for _ in range(max_sample_times):
        if len(sampled_styles) >= num_sample_style:
            break

        style = np.random.choice(QUESTION_TYPES, p=style_probs)
        if style not in sampled_styles:
            sampled_styles.append(style)

    return sampled_styles


def get_dataset_info(examples: List[Dict], sent_limit: int = 100) -> List[Dict]:
    """Extract dataset information from Vietnamese QA examples"""
    examples_with_info = []

    for e in tqdm(examples, desc="Extracting dataset info", unit="example"):
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


def get_sample_probs(examples: List[Dict],
                     answer_length_bins: int = 10,
                     clue_distance_bins: int = 10,
                     answer_length_bin_width: int = 3,
                     answer_length_min_val: int = 0,
                     answer_length_max_val: int = 30,
                     clue_dep_dist_bin_width: int = 2,
                     clue_dep_dist_min_val: int = 0,
                     clue_dep_dist_max_val: int = 20) -> Dict:
    """Calculate sampling probabilities from dataset
    P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
                = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)
    """
    # Get dataset info which includes answer and clue information
    examples_with_info = get_dataset_info(examples)

    # Initialize lists to collect patterns
    sla_tag = []          # for p(s|a_tag)
    clc_tag_dep_dist = []  # for p(c|c_tag, dep_dist)
    ala_tag_a_length = []  # for p(a|a_tag, a_length)

    print("Processing examples for probability calculation...")
    for e in tqdm(examples_with_info):
        try:
            # Get answer info
            a_tag = e["answer_tag"]  # Already in format "POS-NER"

            # Get question style
            s = e["question_type"]

            # Get answer length bin
            a_length = e["answer_length"]
            a_length_bin = val2bin(
                a_length,
                answer_length_min_val,
                answer_length_max_val,
                answer_length_bin_width
            )

            # Get clue and dependency info
            sentence = e["sentence"]
            chunklist, _, tokens = get_chunks(sentence)

            # Find clue in question and its dependency distance
            # You might need to implement this part based on your clue identification logic
            for chunk in chunklist:
                chunk_text = " ".join(chunk[2]).lower()
                if chunk_text in e["question"].lower():
                    c_tag = f"{chunk[1]}-{chunk[0]}"  # POS-NER tag

                    # Calculate dependency distance
                    # Find answer position
                    answer_start = None
                    for i, c in enumerate(chunklist):
                        if " ".join(c[2]) == e["answer_text"]:
                            answer_start = c[3]
                            break

                    if answer_start is not None:
                        # Get dependency distance
                        distance_result = vncore_handler.get_dependency_distance(
                            sentence,
                            chunk[3],  # clue start
                            answer_start,
                            tokens
                        )

                        if distance_result:
                            dep_dist, _ = distance_result
                            dep_dist_bin = val2bin(
                                dep_dist,
                                clue_dep_dist_min_val,
                                clue_dep_dist_max_val,
                                clue_dep_dist_bin_width
                            )

                            # Append patterns
                            sla_tag.append(f"{s}_{a_tag}")
                            clc_tag_dep_dist.append(f"{c_tag}_{dep_dist_bin}")
                            ala_tag_a_length.append(f"{a_tag}_{a_length_bin}")

        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue

    # Convert lists to Counters
    sla_tag = Counter(sla_tag)
    clc_tag_dep_dist = Counter(clc_tag_dep_dist)
    ala_tag_a_length = Counter(ala_tag_a_length)

    # Create final probability dictionary
    sample_probs = {
        "a": ala_tag_a_length,
        "c|a": clc_tag_dep_dist,
        "s|c,a": sla_tag
    }

    # Print statistics
    print("\nProbability Statistics:")
    print(f"Answer patterns (a|tag,length): {len(ala_tag_a_length)}")
    print(f"Clue patterns (c|tag,dist): {len(clc_tag_dep_dist)}")
    print(f"Style patterns (s|a_tag): {len(sla_tag)}")

    return sample_probs


def augment_qg_data(
    sentence: str,
    sample_probs: Dict,
    vncore_handler: VnCorenlpHandler,
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
    print(f"\nProcessing sentence: {sentence}")  # Debug print

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

    print(f"Found {len(chunklist)} chunks")  # Debug print
    for chunk in chunklist:
        # Debug print
        print(f"Chunk: {' '.join(chunk[2])} ({chunk[1]}-{chunk[0]})")

    print(f"Selected {len(sampled_answers)} answers")  # Debug print
    for ans in sampled_answers:
        print(f"Answer: {ans[0]} ({ans[6]}-{ans[7]})")  # Debug print

    sampled_infos = []

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
        print(f"Selected styles for {ans[0]}: {styles}")  # Debug print

        # Sample clues
        clues = select_clues(
            chunklist,
            tokens,
            sample_probs,
            ans,
            vncore_handler,
            num_sample_clue,
            clue_dep_dist_bin_width,
            clue_dep_dist_min_val,
            clue_dep_dist_max_val,
            max_sample_times,
        )
        # Debug print
        print(
            f"Selected clues for {ans[0]}: {[c['clue_text'] for c in clues]}")

        sampled_infos.append(
            {"answer": answer_info, "styles": styles, "clues": clues})

    print(f"Generated {len(sampled_infos)} samples")  # Debug print
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
                vncore_handler=vncore_handler,
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
