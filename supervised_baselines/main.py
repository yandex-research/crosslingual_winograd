# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team., 2019 Intelligent Systems Lab, University of Oxford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
from apex import amp, optimizers
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, \
    get_linear_schedule_with_warmup, XLMRobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from data_reader import DataProcessor
from scorer import scorer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertLMPredictionHeadAlt(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHeadAlt, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHeadAlt(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHeadAlt, self).__init__()
        self.predictions = BertLMPredictionHeadAlt(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    
    The code is taken from pytorch_pretrain_bert/modeling.py, but the loss function has been changed to return
    loss for each example separately.
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHeadAlt(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)

        prediction_scores = self.cls(sequence_output.last_hidden_state)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.permute(0, 2, 1), masked_lm_labels)
            return torch.mean(masked_lm_loss, 1)
        else:
            return prediction_scores


class XLMRobertaForMaskedLMAlt(XLMRobertaForMaskedLM):
    def forward(
            self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.permute(0, 2, 1), masked_lm_labels)
            return torch.mean(masked_lm_loss, 1)
        else:
            return prediction_scores


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, type_1, type_2, masked_lm_1,
                 masked_lm_2):
        self.input_ids_1 = input_ids_1
        self.attention_mask_1 = attention_mask_1
        self.type_1 = type_1
        self.masked_lm_1 = masked_lm_1
        # These are only used for train examples
        self.input_ids_2 = input_ids_2
        self.attention_mask_2 = attention_mask_2
        self.type_2 = type_2
        self.masked_lm_2 = masked_lm_2


def convert_examples_to_features_train(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        # sentence_a = example.text_a.replace('[MASK]', example.candidate_a)
        # sentence_b = example.text_b.replace('[MASK]', example.candidate_b)
        #
        # encoding_1 = tokenizer(sentence_a, padding='max_length', truncation=True, max_length=max_seq_len,
        #                        return_attention_mask=True, )
        # encoding_2 = tokenizer(sentence_b, padding='max_length', truncation=True, max_length=max_seq_len,
        #                        return_attention_mask=True, )

        tokens_sent = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = tokenizer.tokenize(example.candidate_b)
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [], [], [], []
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [], [], [], []
        tokens_1.append(tokenizer.cls_token)
        tokens_2.append(tokenizer.cls_token)
        for token in tokens_sent:
            if "_" in token:
                tokens_1.extend([tokenizer.mask_token for _ in range(len(tokens_a))])
                tokens_2.extend([tokenizer.mask_token for _ in range(len(tokens_b))])
            else:
                tokens_1.append(token)
                tokens_2.append(token)
        tokens_1 = tokens_1[:max_seq_len - 1]  # -1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len - 1]
        if tokens_1[-1] != tokenizer.sep_token:
            tokens_1.append(tokenizer.sep_token)
        if tokens_2[-1] != tokenizer.sep_token:
            tokens_2.append(tokenizer.sep_token)

        type_1 = max_seq_len * [0]  # We do not do any inference.
        type_2 = max_seq_len * [0]  # These embeddings can thus be ignored

        attention_mask_1 = (len(tokens_1) * [1]) + ((max_seq_len - len(tokens_1)) * [0])
        attention_mask_2 = (len(tokens_2) * [1]) + ((max_seq_len - len(tokens_2)) * [0])

        # sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        # replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        for token in tokens_1:
            if token == tokenizer.mask_token:
                if len(input_ids_a) <= 0:
                    raise ValueError(example)  # broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1) < max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token == tokenizer.mask_token:
                if len(input_ids_b) <= 0:
                    raise ValueError(example)  # broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-1)
        while len(masked_lm_2) < max_seq_len:
            masked_lm_2.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(type_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len
        features.append(
            InputFeatures(input_ids_1=input_ids_1,
                          input_ids_2=input_ids_2,
                          attention_mask_1=attention_mask_1,
                          attention_mask_2=attention_mask_2,
                          type_1=type_1,
                          type_2=type_2,
                          masked_lm_1=masked_lm_1,
                          masked_lm_2=masked_lm_2))
    return features


def convert_examples_to_features_evaluate(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_sent = tokenizer.tokenize(example.text_a)

        tokens_1, type_1, attention_mask_1, masked_lm_1 = [], [], [], []
        tokens_1.append(tokenizer.cls_token)
        for token in tokens_sent:
            if "_" in token:
                tokens_1.extend([tokenizer.mask_token for _ in range(len(tokens_a))])
            else:
                tokens_1.append(token)
        tokens_1 = tokens_1[:max_seq_len - 1]  # -1 because of [SEP]
        if tokens_1[-1] != tokenizer.sep_token:
            tokens_1.append(tokenizer.sep_token)

        type_1 = max_seq_len * [0]
        attention_mask_1 = (len(tokens_1) * [1]) + ((max_seq_len - len(tokens_1)) * [0])
        # sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        # replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        for token in tokens_1:
            if token == tokenizer.mask_token:
                if len(input_ids_a) <= 0:
                    continue  # broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1) < max_seq_len:
            masked_lm_1.append(-1)
        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(masked_lm_1) == max_seq_len

        features.append(
            InputFeatures(input_ids_1=input_ids_1,
                          input_ids_2=None,
                          attention_mask_1=attention_mask_1,
                          attention_mask_2=None,
                          type_1=type_1,
                          type_2=None,
                          masked_lm_1=masked_lm_1,
                          masked_lm_2=None))
    return features


@torch.no_grad()
def test(processor, args, tokenizer, model, device, test_set="wscr-test"):
    eval_examples = processor.get_examples(args.data_dir, test_set)
    eval_features = convert_examples_to_features_evaluate(
        eval_examples, args.max_seq_length, tokenizer)
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_segment_ids_1, all_masked_lm_1)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=min(4, cpu_count()))

    model.eval()
    ans_stats = []
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = (tens.to(device) for tens in batch)
        loss = model.forward(input_ids_1, token_type_ids=segment_ids_1, attention_mask=input_mask_1,
                             masked_lm_labels=label_ids_1)

        eval_loss = loss.cpu().numpy()
        for loss in eval_loss:
            curr_id = len(ans_stats)
            ans_stats.append((eval_examples[curr_id].guid, eval_examples[curr_id].ex_true, loss))
    if test_set == "gap-test":
        return scorer(ans_stats, test_set, output_file=os.path.join(args.output_dir, "gap-answers.tsv"))
    elif test_set == "wnli":
        return scorer(ans_stats, test_set, output_file=os.path.join(args.output_dir, "WNLI.tsv"))
    else:
        return scorer(ans_stats, test_set)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--alpha_param",
                        default=10,
                        type=float,
                        help="Discriminative penalty hyper-parameter.")
    parser.add_argument("--beta_param",
                        default=0.4,
                        type=float,
                        help="Discriminative intolerance interval hyper-parameter.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--load_from_file',
                        type=str,
                        default=None,
                        help="Path to the file with a trained model. Default means bert-model is used. Size must match bert-model.")

    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processor = DataProcessor()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_name = "train.txt"

        train_examples = processor.get_examples(args.data_dir, train_name)

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if 'xlm-roberta' in args.model:
        model = XLMRobertaForMaskedLMAlt.from_pretrained(args.model, output_hidden_states=False)
    else:
        model = BertForMaskedLM.from_pretrained(args.model, output_hidden_states=False)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    global_step = 0
    best_acc = 0
    if args.do_train:
        optimizer = optimizers.FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=math.floor(args.warmup_proportion * t_total),
                                                    num_training_steps=t_total)

        model, optimizer = amp.initialize(model, optimizer, opt_level='O2', verbosity=0)

        train_features = convert_examples_to_features_train(
            train_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids_1 = torch.tensor([f.input_ids_1 for f in train_features], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in train_features], dtype=torch.long)
        all_attention_mask_2 = torch.tensor([f.attention_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_1 = torch.tensor([f.type_1 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.type_2 for f in train_features], dtype=torch.long)
        all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in train_features], dtype=torch.long)
        all_masked_lm_2 = torch.tensor([f.masked_lm_2 for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids_1, all_input_ids_2, all_attention_mask_1, all_attention_mask_2,
                                   all_segment_ids_1, all_segment_ids_2, all_masked_lm_1, all_masked_lm_2)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=min(4, cpu_count()))

        improved_epoch = 0
        for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_steps = 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                loss = compute_loss(args, batch, device, model)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            val_acc = test(processor, args, tokenizer, model, device, test_set="valid.txt")
            logger.info(f"Epoch {epoch_num} dev acc {val_acc}")
            if val_acc > best_acc or best_acc == 0.0:
                best_acc = val_acc
                improved_epoch = epoch_num
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))
                with open(os.path.join(args.output_dir, "best_accuracy.txt"), 'w') as f1_report:
                    f1_report.write("{}".format(best_acc))
            elif epoch_num - improved_epoch >= 3:
                logger.info(f'Early stopping at epoch {epoch_num}')
                break

    logger.info("Best dev acc {}".format(best_acc))
    model_dict = torch.load(os.path.join(args.output_dir, "best_model"))
    model.load_state_dict(model_dict)

    if args.do_eval:
        r = test(processor, args, tokenizer, model, device, test_set="train.txt")
        print("train:\t", r)
        logger.info("  >>> train >>> = %f", r)

        r = test(processor, args, tokenizer, model, device, test_set="test.txt")
        print("test:\t", r)
        logger.info("  >>> test >>> = %f", r)

        for lang in 'en', 'jp', 'ru', 'fr', 'zh', 'pt':
            r = test(processor, args, tokenizer, model, device, test_set=f"data_lang_{lang}.txt")
            print(f"{lang}_full:\t", r)
            logger.info(f"  >>> {lang}_full >>> = %f", r)


def compute_loss(args, batch, device, model):
    input_ids_1, input_ids_2, input_mask_1, input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2 = (
        tens.to(device) for tens in batch)
    loss_1 = model.forward(input_ids_1, token_type_ids=segment_ids_1, attention_mask=input_mask_1,
                           masked_lm_labels=label_ids_1)
    loss_2 = model.forward(input_ids_2, token_type_ids=segment_ids_2, attention_mask=input_mask_2,
                           masked_lm_labels=label_ids_2)
    loss = loss_1 + args.alpha_param * torch.max(torch.zeros_like(loss_1),
                                                 torch.ones_like(
                                                     loss_1) * args.beta_param + loss_1 - loss_2.mean())
    loss = loss.mean()
    return loss


if __name__ == "__main__":
    main()
