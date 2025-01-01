"""
Given input Vietnamese sentence, select chunks that can be answer.
Currently, we restrict answer to be a part of input sentence.
"""
from underthesea import chunk, pos_tag, ner
from typing import List, Tuple, Dict
from underthesea import chunk, pos_tag, ner
from .config import INVALID_START_WORDS


def get_ner_tag(text: str, context: str) -> str:
    """Get NER tag for a text span"""
    ner_tag = "UNK"
    ner_results = ner(context)
    for ent in ner_results:
        if text in ent[0]:
            ner_tag = ent[1]
            break
    return ner_tag


def get_token2char(tokens: List[str], text: str) -> Tuple[Dict, Dict]:
    """
    Create mappings between token indices and character positions
    """
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
    """
    Find the character start and end positions of a token sequence in text
    """
    search_text = ' '.join(token_list)
    start_pos = text.find(search_text)

    if start_pos >= 0:
        return start_pos, start_pos + len(search_text) - 1
    return -1, -1


def process_chunks(chunks: List[Tuple], text: str) -> List[Tuple]:
    """
    Process chunks to get valid answer candidates
    Returns only meaningful chunks that could be answers
    """
    answer_chunks = []
    current_np_sequence = []
    current_np_start = None

    for i, chunk_info in enumerate(chunks):
        chunk_text = chunk_info[0]
        chunk_type = chunk_info[1]
        chunk_tag = chunk_info[2]

        if chunk_type == 'CH' or not chunk_text.strip():
            continue

        if chunk_type in ['E', 'R', 'M'] or chunk_text.lower() in INVALID_START_WORDS:
            continue

        if chunk_tag == 'B-NP':
            if current_np_sequence and i > 0 and chunks[i-1][2] == 'B-NP':
                current_np_sequence.append(chunk_text)
            else:
                if current_np_sequence:
                    full_np = ' '.join(current_np_sequence)
                    answer_chunks.append(
                        (full_np, 'NP', get_ner_tag(full_np, text)))
                current_np_sequence = [chunk_text]
                current_np_start = i
        else:
            if current_np_sequence:
                full_np = ' '.join(current_np_sequence)
                answer_chunks.append(
                    (full_np, 'NP', get_ner_tag(full_np, text)))
                current_np_sequence = []

            if chunk_tag == 'B-VP' and len(chunk_text.split()) > 1:
                answer_chunks.append(
                    (chunk_text, 'VP', get_ner_tag(chunk_text, text)))
            elif chunk_tag == 'B-NP':
                answer_chunks.append(
                    (chunk_text, 'NP', get_ner_tag(chunk_text, text)))

    if current_np_sequence:
        full_np = ' '.join(current_np_sequence)
        answer_chunks.append((full_np, 'NP', get_ner_tag(full_np, text)))

    return answer_chunks


def select_answers(context: str, processed=False) -> List[Tuple]:
    """
    Input a Vietnamese context, select which parts can be answers.
    Returns: List of (answer_text, char_start, char_end, bio_ids, label) tuples
    """
    tokens = pos_tag(context)
    token_texts = [token[0] for token in tokens]
    chunks = chunk(context)
    token2idx, idx2token = get_token2char(token_texts, context)
    answer_chunks = process_chunks(chunks, context)

    answers = []
    for chunk_text, chunk_type, ner_tag in answer_chunks:
        try:
            chunk_tokens = chunk_text.split()
            char_start, char_end = str_find(context, chunk_tokens)

            if char_start < 0:
                continue

            token_start = idx2token[char_start]
            token_end = idx2token[char_end]

            bio_ids = ['O'] * len(tokens)
            bio_ids[token_start: token_end + 1] = ['I'] * \
                (token_end - token_start + 1)
            bio_ids[token_start] = 'B'

            final_char_start = token2idx[token_start][0]
            final_char_end = token2idx[token_end][1]

            answers.append((
                chunk_text,
                final_char_start,
                final_char_end,
                bio_ids,
                f"{chunk_type}-{ner_tag}"
            ))

        except:
            continue

    return answers


if __name__ == "__main__":
    test_sentences = [
        "Nguyễn Du là một nhà thơ lớn của Việt Nam.",
        "Hà Nội là thủ đô của nước Việt Nam từ năm 1010.",
        "Trường Đại học Bách Khoa Hà Nội được thành lập vào năm 1956."
    ]

    for sent in test_sentences:
        answer_list = select_answers(sent)
        print("\nSentence:", sent)
        print("Answers:", answer_list)
