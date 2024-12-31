import os
import json
import tqdm
import random
from shutil import move
from argparse import ArgumentParser
from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
from transformers import AutoModelForSeq2SeqLM
from datetime import datetime
import torch
from .dataloader import get_dataset
from .train import SPECIAL_TOKENS
from metrics.text_generation_metrics import compute_metrics_by_file


def build_encoder_input(inst, tokenizer):
    """Build encoder input matching BartPho training format"""
    paragraph, clue, style, answer, question = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])

    # Build encoder inputs exactly as in training
    encoder_inputs = []

    # Add paragraph
    encoder_inputs.extend([paragraph] + inst['paragraph'])

    # Add answer
    encoder_inputs.extend([answer] + inst['answer'])

    # Add clue if exists
    if inst['clue_start'] is not None:
        encoder_inputs.extend([clue] + inst['clue'])

    # Add style
    encoder_inputs.extend([style] + inst['style'])

    # Create attention mask
    encoder_attention_mask = [1] * len(encoder_inputs)

    return {
        "input_ids": encoder_inputs,
        "attention_mask": encoder_attention_mask
    }


def generate_question(model, inst, tokenizer, args):
    """Generate question using BartPho training format"""
    inst['original_question'] = inst.get('question', [])

    # Get encoder inputs
    encoder_input = build_encoder_input(inst, tokenizer)
    input_ids = torch.tensor([encoder_input["input_ids"]], device=args.device)
    attention_mask = torch.tensor(
        [encoder_input["attention_mask"]], device=args.device)

    # Initialize decoder with correct decoder_start_token_id
    # Use decoder_start_token_id=2
    decoder_input_ids = torch.tensor([[2]], device=args.device)

    output_sequence = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=5,
        do_sample=False,  # Disable sampling for more stable output
        top_k=50,
        temperature=1.0,
        repetition_penalty=1.5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        pad_token_id=1,  # Use correct pad_token_id
        eos_token_id=2,  # Use correct eos_token_id
        forced_eos_token_id=2,  # Force correct EOS token
        early_stopping=True
    )

    return output_sequence[0].tolist(), inst['original_question']


def run():
    parser = ArgumentParser()
    # Basic parameters
    parser.add_argument("--model_type", type=str,
                        default="vinai/bartpho-syllable", help="Model type")
    parser.add_argument("--model_name_or_path", type=str,
                        required=True, help="Path to the model")
    parser.add_argument("--filename", type=str,
                        required=True, help="Input file")
    parser.add_argument("--filecache", type=str,
                        required=True, help="Cache file")
    parser.add_argument("--output_file", type=str,
                        required=True, help="Output file")

    # Generation parameters
    parser.add_argument("--no_sample", action='store_true',
                        help="Use greedy decoding instead of sampling")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) filtering")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="Size of n-grams to prevent repetition")
    parser.add_argument("--min_length", type=int, default=5,
                        help="Minimum length of output")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of output")

    # Other parameters
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--debug", action='store_true',
                        help="Debug mode")
    parser.add_argument("--save_freq", type=int, default=2000,
                        help="Save frequency")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Initialize model and tokenizer
    print("Initializing model and tokenizer")
    tokenizer = BartphoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    model.eval()

    # Load data
    print("Loading dataset")
    data = get_dataset(tokenizer, args.filecache,
                       args.filename, debug=args.debug)

    # Initialize output dictionary
    final_output_dict = {
        "version": "fqg",
        "data": [{
            "paragraphs": []
        }]
    }

    # Process instances
    question_number = 0
    processed_para_indexs = []
    para_index2output_dict_paras_index = {}

    start = datetime.now()
    for inst in tqdm.tqdm(data):
        try:
            with torch.no_grad():
                generated_question, original_question_ids = generate_question(
                    model, inst, tokenizer, args)

            # Decode outputs
            original_paragraph = tokenizer.decode(inst['paragraph'])
            original_answer = tokenizer.decode(
                inst['answer'], skip_special_tokens=True)
            generated_question_text = tokenizer.decode(
                generated_question, skip_special_tokens=True)
            original_question_text = tokenizer.decode(
                original_question_ids, skip_special_tokens=True)
            para_index = inst['para_id']

            # Verify answer position
            original_ans_position = inst["answer_position"]
            if original_paragraph[original_ans_position:original_ans_position + len(original_answer)] != original_answer:
                print("Answer position mismatch, attempting to fix...")
                new_answer_position = original_paragraph.find(original_answer)
                if new_answer_position >= 0:
                    inst["answer_position"] = new_answer_position
                    original_ans_position = new_answer_position
                else:
                    continue

            if para_index in processed_para_indexs:
                output_para_index = para_index2output_dict_paras_index[para_index]
                final_output_dict["data"][0]["paragraphs"][output_para_index]['qas'].append({
                    'id': f'question_{question_number}',
                    'question': generated_question_text,
                    'original_question': original_question_text,
                    'answers': [{
                        'text': original_answer,
                        'answer_start': original_ans_position,
                    }],
                    'clue': tokenizer.decode(inst["clue"]),
                    'clue_start': inst["clue_start"],
                    'ques_type': inst['ques_type'],
                    'ques_type_text': tokenizer.decode(inst["style"])
                })
            else:
                final_output_dict['data'][0]['paragraphs'].append({
                    'context': original_paragraph,
                    'qas': [{
                        'id': f'question_{question_number}',
                        'question': generated_question_text,
                        'original_question': original_question_text,
                        'answers': [{
                            'text': original_answer,
                            'answer_start': original_ans_position,
                        }],
                        'clue': tokenizer.decode(inst["clue"]),
                        'clue_start': inst["clue_start"],
                        'ques_type': inst['ques_type'],
                        'ques_type_text': tokenizer.decode(inst["style"])
                    }]
                })
                processed_para_indexs.append(para_index)
                para_index2output_dict_paras_index[para_index] = len(
                    processed_para_indexs) - 1

            question_number += 1

            # Save intermediate results
            if len(processed_para_indexs) % args.save_freq == 0:
                if os.path.isfile(args.output_file):
                    move(args.output_file, args.output_file + ".copy.json")
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(final_output_dict, f,
                              indent=4, ensure_ascii=False)
                    print(f"Saved {question_number} generated questions.")

        except Exception as e:
            print(f"Error processing instance: {e}")
            continue

    print(
        f"Time to generate {question_number} questions: {datetime.now() - start}")

    # Save final results
    if os.path.isfile(args.output_file):
        move(args.output_file, args.output_file + ".copy.json")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    run()
