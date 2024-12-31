import os
import math
from pprint import pformat
from .arguments import *
from .dataloader import get_dataset
from collections import defaultdict
from utils.utils import save_result_to_drive

import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
from transformers import AutoModelForSeq2SeqLM

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear

SPECIAL_TOKENS = ['<sos>', '<eos>', '<paragraph>',
                  '<clue>', '<style>', '<answer>', '<question>', '<pad>']
MODEL_INPUTS = ["input_ids", "attention_mask",
                "decoder_input_ids", "decoder_attention_mask", "labels"]
CHECKPOINT_PREFIX = 'checkpoint'

logger = logging.getLogger(__file__)


def pad_dataset(dataset, dataset_name, padding=0):
    """Pad the dataset with appropriate padding for encoder and decoder inputs"""
    max_encoder_l = max(len(x) for x in dataset["input_ids"])
    max_decoder_l = max(len(x) for x in dataset["decoder_input_ids"])

    for name in MODEL_INPUTS:
        if name in ["input_ids", "attention_mask"]:
            # Pad encoder inputs
            dataset[name] = [x + [padding if name == "input_ids" else 0] * (max_encoder_l - len(x))
                             for x in dataset[name]]
        elif name in ["decoder_input_ids", "decoder_attention_mask", "labels"]:
            # Pad decoder inputs
            dataset[name] = [x + [padding if name == "decoder_input_ids" else 0 if name == "decoder_attention_mask" else -100] * (max_decoder_l - len(x))
                             for x in dataset[name]]
    return dataset


def build_input_from_segments(data_point, tokenizer, dataset_name, with_eos=True):
    """Build encoder and decoder inputs"""
    # Get special token ids
    sos, eos, paragraph, clue, style, answer, question = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])

    # Prepare encoder inputs (source)
    encoder_inputs = []
    # Add paragraph
    encoder_inputs.extend([paragraph] + data_point['paragraph'])

    # Add answer
    encoder_inputs.extend([answer] + data_point['answer'])

    # Add clue if exists
    if data_point['clue_start'] is not None:
        encoder_inputs.extend([clue] + data_point['clue'])

    # Add style
    encoder_inputs.extend([style] + data_point['style'])

    # Prepare decoder inputs (target)
    decoder_inputs = [sos]  # Start with SOS token
    decoder_inputs.extend(data_point['question'])
    if with_eos:
        decoder_inputs.append(eos)

    # Create attention masks
    encoder_attention_mask = [1] * len(encoder_inputs)
    decoder_attention_mask = [1] * len(decoder_inputs)

    # Labels are the decoder inputs shifted right
    labels = decoder_inputs[1:]  # Remove SOS token

    instance = {
        "input_ids": encoder_inputs,
        "attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_inputs,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels
    }

    return instance, encoder_inputs


def get_data_loaders(tokenizer, args):
    """Create train and validation data loaders"""
    datasets_raw = {}
    logger.info("Loading training data")
    datasets_raw['train'] = get_dataset(
        tokenizer, args.train_dataset_cache_path, args.train_dataset_path, args.debug)
    logger.info("Loading validation data")
    datasets_raw['valid'] = get_dataset(
        tokenizer, args.dev_dataset_cache_path, args.dev_dataset_path, args.debug)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in datasets_raw.items():
        for data_point in dataset:
            instance, _ = build_input_from_segments(
                data_point, tokenizer, dataset_name)
            for input_name, input_array in instance.items():
                datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset,
            dataset_name,
            padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        )
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(
        *tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    return train_loader, valid_loader


def train():
    args = parser.parse_args()

    # Initialize tokenizer and model
    tokenizer = BartphoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(tokenizer, args)

    # Define training function
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    # Define evaluation function
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )

            lm_logits = outputs.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            return shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)

    def upload_data_to_drive(engine):
        epoch = engine.state.epoch
        score = engine.state.metrics
        checkpoint_file_path = f'{args.output_dir}/{CHECKPOINT_PREFIX}_epoch_{epoch}.pt'
        save_result_to_drive(epoch, args.prefix,
                             args.output_dir, score, checkpoint_file_path)

    # Create trainer and evaluator
    trainer = Engine(update)
    evaluator = Engine(inference)

    # Attach evaluation to trainer
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(
            Events.STARTED, lambda _: evaluator.run(val_loader))

    # Learning rate scheduler
    scheduler = PiecewiseLinear(
        optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100))}
    metrics["ppl"] = MetricsLambda(math.exp, metrics["nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # Progress bar and checkpoints
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(
            Events.COMPLETED,
            lambda _: pbar.log_message(
                "Validation: %s" % pformat(evaluator.state.metrics))
        )

        checkpoint_handler = ModelCheckpoint(
            args.output_dir,
            filename_prefix=CHECKPOINT_PREFIX,
            filename_pattern='{filename_prefix}_epoch_{global_step}.{ext}',
            global_step_transform=lambda e, _: e.state.epoch,
            n_saved=3,
            require_empty=False
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handler,
            {'mymodel': getattr(model, 'module', model)}
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, upload_data_to_drive)

        # Save model and tokenizer
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Run training
    trainer.run(train_loader, max_epochs=args.n_epochs)


if __name__ == "__main__":
    train()
