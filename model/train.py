import os
import math
from pprint import pformat
from .arguments import *
from .dataloader import get_dataset
from collections import defaultdict

import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import WEIGHTS_NAME, CONFIG_NAME

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear


SPECIAL_TOKENS = ['<sos>', '<eos>', '<paragraph>', '<clue>', '<style>', '<answer>', '<question>', '<pad>']
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

def pad_dataset(dataset, dataset_name, padding=0, ):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    masked_value = -100 if dataset_name == 'train' else -1
    for name in MODEL_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else masked_value] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(data_point, tokenizer, dataset_name, with_eos=True):
  """ Build a sequence of input.
      `<sos> .. paragraph text ..
      <clue> .. clue span ..
      <answer> .. answer span ..
      <style> .. question style ..
      <question> .. question span .. <eos>`
  """
  sos, eos, paragraph, clue, style, answer, question = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

  masked_value = -100 if dataset_name == 'train' else -1
  curr_para = data_point['paragraph']
  curr_ans = data_point['answer']
  curr_ques = data_point['question']
  ans_start = data_point['answer_position_tokenized'][0]
  ans_end = data_point['answer_position_tokenized'][1]
  curr_style = data_point['style']

  clue_exist = (data_point['clue_start'] is not None)
  if clue_exist:
      curr_clue = data_point['clue']
      clue_start = data_point['clue_position_tokenized'][0]
      clue_end = data_point['clue_position_tokenized'][1]
  else:
      curr_clue = []

  # <sos> paragraph
  sequence = [sos] + curr_para
  # This segmentation will encode positional information
  token_types = [paragraph for i in range(len(curr_para) + 1)]
  if clue_exist:
      token_types[clue_start + 1:clue_end + 1] = [clue] * (clue_end - clue_start)
  token_types[ans_start + 1:ans_end + 1] = [answer] * (ans_end - ans_start)
  lm_labels = [masked_value for _ in range(len(curr_para) + 1)]

  # <sos> paragraph <answer> answer
  sequence.extend([answer] + curr_ans)
  token_types.extend([answer for _ in range(len(curr_ans) + 1)])
  lm_labels.extend([masked_value for _ in range(len(curr_ans) + 1)])

  # <sos> paragraph <answer> answer <clue> clue
  sequence.extend([clue] + curr_clue)
  token_types.extend([clue for _ in range(len(curr_clue) + 1)])
  lm_labels.extend([masked_value for _ in range(len(curr_clue) + 1)])

  # <sos> paragraph <answer> answer <clue> clue <style> style
  sequence.extend([style] + curr_style)
  token_types.extend([style for _ in range(len(curr_style) + 1)])
  lm_labels.extend([masked_value for _ in range(len(curr_style) + 1)])

  # <sos> paragraph <answer> answer <clue> clue <style> style <question> question [<eos>]
  if with_eos is True:
      sequence.extend([question] + curr_ques + [eos])
      token_types.extend([question for _ in range(len(curr_ques) + 2)])
      lm_labels.extend([masked_value] + curr_ques + [eos])
  else:
      sequence.extend([question] + curr_ques)
      token_types.extend([question for _ in range(len(curr_ques) + 1)])
      lm_labels.extend([masked_value] + curr_ques)

  assert len(sequence) == len(token_types)
  assert len(token_types) == len(lm_labels)

  instance = {
      "input_ids": sequence,
      "token_type_ids": token_types,
      "lm_labels": lm_labels
  }
  return instance, sequence

def get_data_loaders(tokenizer, args):
  datasets_raw = {}
  logger.info("Loading training data")
  datasets_raw['train'] = get_dataset(tokenizer, args.train_dataset_cache_path, args.train_dataset_path, 'train', args.debug)
  logger.info("Loading validation data")
  datasets_raw['valid'] = get_dataset(tokenizer, args.dev_dataset_cache_path, args.dev_dataset_path, 'dev', args.debug)
  
  
  logger.info("Build inputs and labels")
  datasets = {
      "train": defaultdict(list),
      "valid": defaultdict(list)
  }
  
  for dataset_name, dataset in datasets_raw.items():
    for data_point in dataset:
      instance, _ = build_input_from_segments(data_point, tokenizer, dataset_name)
      for input_name, input_array in instance.items():
        datasets[dataset_name][input_name].append(input_array)

  logger.info("Pad inputs and convert to Tensor")
  tensor_datasets = {"train": [], "valid": []}
  for dataset_name, dataset in datasets.items():
      dataset = pad_dataset(dataset, dataset_name, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
      for input_name in MODEL_INPUTS:
        tensor = torch.tensor(dataset[input_name])
        tensor_datasets[dataset_name].append(tensor)

  logger.info("Build train and validation dataloaders")
  train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
  train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
  valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

  logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
  logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
  return train_loader, valid_loader

def train():
  args = parser.parse_args()
  tokenizer = BartphoTokenizer.from_pretrained(args.model_name_or_path)
  tokenizer.add_tokens(SPECIAL_TOKENS)
  model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
  model.to(args.device)
  optimizer = AdamW(model.parameters(), lr=args.lr)
  
  train_loader, val_loader = get_data_loaders(tokenizer, args)
  # Training function and trainer
  def update(engine, batch):
    model.train()
    inputs = tuple(input_tensor.to(args.device) for input_tensor in batch) 
    cur_input_ids, cur_lm_labels, cur_token_type_ids = inputs
    outputs = model(input_ids=cur_input_ids, token_type_ids=cur_token_type_ids, labels=cur_lm_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
  trainer = Engine(update)
  
  # Evaluation function and evaluator (evaluator output is the input of the metrics)
  def inference(engine, batch):
    model.eval()
    with torch.no_grad():
      batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
      input_ids, lm_labels, token_type_ids = batch
      
      model_outputs = model(input_ids, token_type_ids=token_type_ids)
      lm_logits = model_outputs.logits
      
      lm_logits_flat = lm_logits.contiguous().view(-1, lm_logits.size(-1))
      lm_labels_flat = lm_labels.contiguous().view(-1)

      return lm_logits_flat, lm_labels_flat
  evaluator = Engine(inference)
    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
  trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
  if args.n_epochs < 1:
      trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
  if args.eval_before_start:
      trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

  # Linearly decrease the learning rate from lr to zero
  scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
  trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

  # Prepare metrics - note how we compute distributed metrics
  RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
  metrics = {
      "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1))
  }
  metrics["ppl"] = MetricsLambda(math.exp, metrics["nll"])
  for name, metric in metrics.items():
      metric.attach(evaluator, name)

  # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
  pbar = ProgressBar(persist=True)
  pbar.attach(trainer, metric_names=["loss"])
  evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

  checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=None, n_saved=3)  # !!!NOTICE: if fill exist, it will report error. set require_empty=False can avoid this.
  trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

  model.save_pretrained(args.output_dir)

  # torch.save(args, args.output_dir + '/model_training_args.bin')
  getattr(model, 'module', model).config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
  tokenizer.save_vocabulary(args.output_dir)
  
  if args.n_epochs > 0:
    os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.output_dir, WEIGHTS_NAME)) 

  # Run the training
  trainer.run(train_loader, max_epochs=args.n_epochs)
  
  
if __name__ == "__main__":
    train()