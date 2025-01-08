# create_sample_probs.py

import json
import logging
from tqdm import tqdm
from typing import List, Dict
import pickle
import os

from data_agumentor.data_augmentor import get_sample_probs, get_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_viquad_data(file_path: str) -> List[Dict]:
    """Load and process ViQuAD data into the format we need"""
    logger.info(f"Loading ViQuAD data from {file_path}")

    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each article
        for article in tqdm(data['data'], desc="Processing ViQuAD articles"):
            for paragraph in article['paragraphs']:
                context = paragraph['context']

                # Process each QA pair
                for qa in paragraph['qas']:
                    # Get the answer
                    if not qa['answers']:
                        continue

                    answer = qa['answers'][0]  # Take first answer

                    # Get the sentence containing the answer
                    ans_start = answer['answer_start']
                    ans_text = answer['text']

                    # Extract the sentence containing the answer
                    sentences = context.split('.')  # Simple sentence splitting
                    current_pos = 0
                    ans_sent = None
                    ans_sent_start = 0

                    for sent in sentences:
                        sent = sent.strip() + '.'
                        sent_length = len(sent)

                        if current_pos <= ans_start < current_pos + sent_length:
                            ans_sent = sent
                            ans_sent_start = current_pos
                            break

                        current_pos += sent_length + 1  # +1 for the space

                    if ans_sent:
                        # Adjust answer start position relative to sentence
                        relative_start = ans_start - ans_sent_start

                        examples.append({
                            "ans_sent": ans_sent,
                            "question": qa['question'],
                            "answer_text": ans_text,
                            "answer_start": relative_start,
                            "id": qa['id']
                        })

        logger.info(f"Loaded {len(examples)} examples from ViQuAD")
        return examples

    except Exception as e:
        logger.error(f"Error loading ViQuAD data: {str(e)}")
        raise


def create_sample_probs_from_viquad(viquad_file: str,
                                    output_file: str = "viquad_sample_probs.pkl",
                                    answer_length_bins: int = 10,
                                    clue_distance_bins: int = 10):
    """Create sampling probabilities file from ViQuAD dataset"""
    try:
        # Load ViQuAD data
        examples = load_viquad_data(viquad_file)

        if not examples:
            raise ValueError("No examples loaded from ViQuAD")

        # Calculate probabilities
        logger.info("Calculating sampling probabilities...")
        sample_probs = get_sample_probs(
            examples,
            answer_length_bins=answer_length_bins,
            clue_distance_bins=clue_distance_bins
        )

        # Save probabilities
        logger.info(f"Saving probabilities to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(sample_probs, f)

        return sample_probs

    except Exception as e:
        logger.error(f"Error creating sample probabilities: {str(e)}")
        raise


def test_sample_probs(sample_probs: Dict):
    """Test the generated sample probabilities"""
    logger.info("\nSample probabilities statistics:")
    logger.info(f"Answer probabilities: {len(sample_probs['a'])} entries")
    logger.info(f"Clue probabilities: {len(sample_probs['c|a'])} entries")
    logger.info(f"Style probabilities: {len(sample_probs['s|c,a'])} entries")

    # Print some example probabilities
    logger.info("\nExample probabilities:")
    for key in sample_probs:
        logger.info(f"\n{key}:")
        for prob_key, prob_value in list(sample_probs[key].items())[:5]:
            logger.info(f"{prob_key}: {prob_value}")


def main():
    """Main function to create sample probabilities"""
    # Paths
    viquad_train_file = "datasets/ViQuAD1.0/train_ViQuAD.json"
    output_file = "viquad_sample_probs.pkl"

    try:
        # Create sample probabilities
        logger.info("Starting sample probability creation from ViQuAD...")
        sample_probs = create_sample_probs_from_viquad(
            viquad_file=viquad_train_file,
            output_file=output_file
        )

        # Test the probabilities
        test_sample_probs(sample_probs)

        logger.info("\nSample probabilities creation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
