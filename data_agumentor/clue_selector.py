"""
Select the chunks that can be clue to given context and answer for Vietnamese text.
"""
from typing import List, Tuple, Dict
from underthesea import chunk, pos_tag, ner
from .config import INVALID_POS_TAGS, DISTANCE_THRESHOLD


def get_chunks_info(context: str):
    """
    Get chunks and their information from Vietnamese text
    """
    # Get tokens and chunks
    tokens = pos_tag(context)
    chunks = chunk(context)

    # Create token and chunk mappings
    token_texts = [t[0] for t in tokens]
    token_pos = [t[1] for t in tokens]

    chunk_info = []
    current_pos = 0

    for chunk_text, chunk_type, chunk_tag in chunks:
        # Find chunk position
        chunk_start = -1
        for i in range(current_pos, len(token_texts)):
            if chunk_text.startswith(token_texts[i]):
                chunk_start = i
                break

        if chunk_start != -1:
            chunk_tokens = chunk_text.split()
            chunk_end = chunk_start + len(chunk_tokens) - 1
            current_pos = chunk_end + 1

            chunk_info.append({
                "text": chunk_text,
                "type": chunk_type,
                "tag": chunk_tag,
                "start": chunk_start,
                "end": chunk_end
            })

    return token_texts, token_pos, chunk_info


def is_valid_clue(chunk_info: Dict, token_pos: List[str]) -> bool:
    """
    Check if a chunk can be a valid clue
    """
    # Skip chunks with invalid POS tags
    chunk_start_pos = token_pos[chunk_info["start"]]
    if chunk_start_pos in INVALID_POS_TAGS:
        return False

    # Skip single tokens that are function words
    if chunk_info["start"] == chunk_info["end"] and chunk_start_pos in ['E', 'C', 'T', 'M']:
        return False

    # Skip chunks that are too short
    if len(chunk_info["text"].split()) < 2:
        return False

    return True


def select_clues(context: str, answer: str, answer_bio_ids: List[str],
                 max_distance: int = DISTANCE_THRESHOLD) -> List[Dict]:
    """
    Select clue chunks given Vietnamese context and answer.
    Returns: List of {"clue_text": str, "clue_binary_ids": List[int]} dictionaries
    """
    # Get tokens and chunks information
    token_texts, token_pos, chunk_info = get_chunks_info(context)

    # Find answer span
    answer_start = answer_bio_ids.index('B')
    try:
        answer_end = len(answer_bio_ids) - 1 - \
            list(reversed(answer_bio_ids)).index('I')
    except:
        answer_end = answer_start

    clues = []
    for chunk in chunk_info:
        # Skip the answer chunk itself
        if chunk["start"] >= answer_start and chunk["end"] <= answer_end:
            continue

        # Check if chunk is valid and within distance threshold
        if is_valid_clue(chunk, token_pos):
            # Calculate distance to answer
            distance_to_answer = min(
                abs(chunk["start"] - answer_end),
                abs(chunk["end"] - answer_start)
            )

            if distance_to_answer <= max_distance:
                # Create binary IDs for the clue
                binary_ids = [0] * len(token_texts)
                binary_ids[chunk["start"]:chunk["end"] + 1] = [1] * \
                    (chunk["end"] - chunk["start"] + 1)

                clues.append({
                    "clue_text": chunk["text"],
                    "clue_binary_ids": binary_ids
                })

    return clues


if __name__ == "__main__":
    # Test examples
    test_cases = [
        {
            "context": "Nguyễn Du là một nhà thơ lớn của Việt Nam",
            "answer": "Nguyễn Du",
            "answer_bio_ids": ["B", "I", "O", "O", "O", "O", "O", "O", "O"],
        },
        {
            "context": "Hà Nội là thủ đô của nước Việt Nam từ năm 1010",
            "answer": "Hà Nội",
            "answer_bio_ids": ["B", "I", "O", "O", "O", "O", "O", "O", "O", "O"],
        },
        {
            "context": "Trường Đại học Bách Khoa Hà Nội được thành lập vào năm 1956",
            "answer": "Trường Đại học Bách Khoa Hà Nội",
            "answer_bio_ids": ["B", "I", "I", "I", "I", "I", "O", "O", "O", "O", "O", "O"],
        }
    ]

    for test in test_cases:
        print("\nContext:", test["context"])
        print("Answer:", test["answer"])
        clues = select_clues(
            test["context"],
            test["answer"],
            test["answer_bio_ids"]
        )
        print("Clues:", clues)
