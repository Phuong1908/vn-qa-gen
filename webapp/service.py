from model.interact import generate_question
from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
from transformers import AutoModelForSeq2SeqLM
from data_agumentor.data_augmentor import (
    augment_qg_data
)
from data_agumentor.config import QUESTION_PATTERNS
from typing import Dict
import torch
import pickle
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # This will override any existing logger configuration
)
logger = logging.getLogger(__name__)


class QAGenerationService:
    def __init__(self, model_path):
        # Initialize model and tokenizer
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartphoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

        # Load or initialize sampling probabilities
        self.sample_probs = self.load_sample_probs()

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

    def prepare_instance(self, paragraph: str, answer_info: Dict, style: str, clue: Dict) -> Dict:
        """Prepare instance for question generation with proper tokenization"""
        # Tokenize all inputs
        tokenized_paragraph = self.tokenizer.encode(
            paragraph,
            add_special_tokens=False,
            return_tensors=None
        )

        tokenized_answer = self.tokenizer.encode(
            answer_info['answer_text'],
            add_special_tokens=False,
            return_tensors=None
        )

        tokenized_clue = self.tokenizer.encode(
            clue['clue_text'],
            add_special_tokens=False,
            return_tensors=None
        ) if clue['clue_text'] else []

        # Get patterns for the style and randomly select one
        if style in QUESTION_PATTERNS:
            patterns = QUESTION_PATTERNS[style]
            selected_pattern = np.random.choice(patterns)
        else:
            selected_pattern = "gì"  # Default fallback

        selected_pattern = '<mask>' if selected_pattern == "other" else selected_pattern

        # Tokenize the selected pattern
        tokenized_style = self.tokenizer.encode(
            selected_pattern,
            add_special_tokens=False,
            return_tensors=None
        )

        clue_start = -1
        if clue['clue_text']:
            # Simple approach: find first occurrence of clue text in paragraph
            clue_pos = paragraph.find(clue['clue_text'])
            if clue_pos >= 0:
                # Count tokens up to this position
                prefix = paragraph[:clue_pos]
                clue_start = len(self.tokenizer.encode(
                    prefix, add_special_tokens=False))

        return {
            'paragraph': tokenized_paragraph,
            'answer': tokenized_answer,
            'answer_position': answer_info['char_start'],
            'clue_start': clue_start,
            'clue': tokenized_clue,
            'style': tokenized_style,
            'para_id': 0
        }

    def process_paragraph(self, paragraph):
        try:
            logger.info(f"Processing paragraph: {paragraph}")

            # 1. Data Augmentation
            augmented_data = augment_qg_data(
                sentence=paragraph,
                sample_probs=self.sample_probs,
                num_sample_answer=3,
                num_sample_clue=2,
                num_sample_style=2
            )

            logger.info(
                f"Augmented data samples: {len(augmented_data['samples'])}")

            if not augmented_data['samples']:
                logger.warning("No samples generated during augmentation")
                return []

            generated_qa_pairs = []

            # 2. Generate questions for each sample
            for i, sample in enumerate(augmented_data['samples']):
                answer_info = sample['answer']

                # Skip invalid answers
                if not self.is_valid_answer(answer_info['answer_text']):
                    logger.info(
                        f"Skipping invalid answer: {answer_info['answer_text']}")
                    continue

                for style in sample['styles']:
                    clues = sample.get('clues', [{'clue_text': ''}])

                    for clue in clues:
                        # Prepare input with tokenization
                        inst = self.prepare_instance(
                            paragraph=paragraph,
                            answer_info=answer_info,
                            style=style,
                            clue=clue
                        )

                        try:
                            # Generate question
                            with torch.no_grad():
                                generated_question, _ = generate_question(
                                    model=self.model,
                                    inst=inst,
                                    tokenizer=self.tokenizer,
                                    device=self.device  # Pass the device object
                                )

                            # Decode question
                            question = self.tokenizer.decode(
                                generated_question,
                                skip_special_tokens=True
                            )

                            # Add to results if question is generated
                            if question:
                                qa_pair = {
                                    'question': question,
                                    'answer': answer_info['answer_text'],
                                    'style': style,
                                    'clue': clue['clue_text']
                                }
                                generated_qa_pairs.append(qa_pair)
                                logger.info(f"Generated QA pair: {qa_pair}")

                        except Exception as e:
                            logger.error(
                                f"Error generating question: {str(e)}")
                            continue

            logger.info(f"Total generated QA pairs: {len(generated_qa_pairs)}")
            return generated_qa_pairs

        except Exception as e:
            logger.error(
                f"Error in question generation: {str(e)}", exc_info=True)
            raise Exception(f"Error in question generation: {str(e)}")

    def is_valid_answer(self, answer_text: str) -> bool:
        """Check if an answer is valid for question generation"""
        # Skip punctuation-only answers
        if answer_text in ['.', ',', '!', '?', ':', ';']:
            return False

        # Skip very short answers
        if len(answer_text.strip()) < 2:
            return False

        # Skip answers that are just function words
        if answer_text.lower() in ['là', 'và', 'của', 'với']:
            return False

        return True
