import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from nltk import word_tokenize
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaForMaskedLM

from utils import (clean_sentence, tokenize, locate_reference)

softmax = torch.nn.Softmax(dim=-1)

MODEL_CLASSES = {
    'bert-base-multilingual-uncased': (BertTokenizer, BertForMaskedLM, '[MASK]', '##'),
    'xlm-roberta-large': (XLMRobertaTokenizer, XLMRobertaForMaskedLM, '<mask>', '▁'),
}


def base1(args, model, tokenizer, sent_tokens, loc, right_answer_tokens):
    unmasked = sent_tokens[:loc] + list(right_answer_tokens) + sent_tokens[loc + 1:]

    masked = sent_tokens[:loc] + [args.mask_token, ] * len(right_answer_tokens) + sent_tokens[loc + 1:]
    masked_input_ids = tokenizer.convert_tokens_to_ids(masked)
    masked_input_ids = torch.tensor([masked_input_ids, ], dtype=torch.long).to(args.device)

    output = model(masked_input_ids, output_attentions=False)
    probs = softmax(output.get('logits', output.logits))

    score = 0.
    for pos in range(len(unmasked)):
        if pos in (0, len(unmasked) - 1):
            continue
        p = float(probs[0][pos].cpu().detach().numpy()[args.vocab[unmasked[pos]]])
        score += np.log(1e-10 + p)
    return score


def base2(args, model, tokenizer, sent_tokens, loc, right_answer_tokens):
    unmasked = sent_tokens[:loc] + list(right_answer_tokens) + sent_tokens[loc + 1:]
    unmasked_input_ids = tokenizer.convert_tokens_to_ids(unmasked)
    unmasked_input_ids = torch.tensor([unmasked_input_ids, ], dtype=torch.long).to(args.device)

    output = model(unmasked_input_ids, output_attentions=False)
    probs = softmax(output.get('logits', output.logits))

    score = 0.
    for pos in range(len(unmasked)):
        if pos in (0, len(unmasked) - 1):
            continue
        p = float(probs[0][pos].cpu().detach().numpy()[args.vocab[unmasked[pos]]])
        score += np.log(1e-10 + p)
    return score


def base3(args, model, tokenizer, sent_tokens, loc, right_answer_tokens):
    unmasked = sent_tokens[:loc] + list(right_answer_tokens) + sent_tokens[loc + 1:]

    score = 0.
    for pos in range(len(unmasked)):
        if pos in (0, len(unmasked) - 1):
            continue

        masked = unmasked[:]
        masked[pos] = args.mask_token
        masked_input_ids = tokenizer.convert_tokens_to_ids(masked)
        masked_input_ids = torch.tensor([masked_input_ids, ], dtype=torch.long).to(args.device)

        output = model(masked_input_ids, output_attentions=False)
        probs = softmax(output.get('logits', output.logits))

        p = float(probs[0][pos].cpu().detach().numpy()[args.vocab[unmasked[pos]]])
        score += np.log(1e-10 + p)
    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default='xlm-roberta-large',
                        help="Modelname selected in the list: " + ", ".join(list(MODEL_CLASSES.keys())))
    parser.add_argument("--input_file", default='dataset.v5.tsv', type=str,  # required=True,
                        help="The input .tsv file.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    assert args.model in MODEL_CLASSES, 'Unknown model!'

    tokenizer_class, model_class, mask_token, token_prefix = MODEL_CLASSES.get(args.model)

    args.mask_token = mask_token
    args.token_prefix = token_prefix

    tokenizer = tokenizer_class.from_pretrained(args.model, do_lower_case=True)
    model = model_class.from_pretrained(args.model, output_attentions=True)
    model.to(args.device)

    args.vocab = tokenizer.get_vocab()

    stats = defaultdict(lambda: defaultdict(int))

    with open(args.input_file, encoding='utf-8') as ifh:
        for line in ifh:
            chunks = line.strip().split('\t')

            lang = chunks[0]
            ref = json.loads(chunks[5])
            answers = json.loads(chunks[6])
            sentence = clean_sentence(chunks[3])

            sent_input_ids, sent_tokens = tokenize(tokenizer, sentence)

            nltk_sent_tokens = json.loads(chunks[4])

            # deal with reference tokens location, despite artefacts of given tokenizer
            ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence)
            if not ref_locations:
                ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence,
                                                             method='patch')
            if not ref_locations:
                ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence,
                                                             method='approx')
            assert len(ref_locations), 'The reference not found in the sent_tokens.'

            decoys = []
            right_answer = None
            for answer in answers:
                if answer[-1]:
                    right_answer = answer
                else:
                    decoys.append(answer)
            assert right_answer is not None, 'Right answer not found.'

            _, right_answer_tokens = tokenize(tokenizer, right_answer[0], False)
            right_answer_tokens_denom1 = len(sent_tokens) - 1 + len(right_answer_tokens) - 2
            right_answer_tokens_denom2 = len(nltk_sent_tokens) - len(word_tokenize(right_answer[0]))
            right_answer_raw_score1 = base1(args, model, tokenizer, sent_tokens, ref_locations[0], right_answer_tokens)
            right_answer_raw_score2 = base2(args, model, tokenizer, sent_tokens, ref_locations[0], right_answer_tokens)
            right_answer_raw_score3 = base3(args, model, tokenizer, sent_tokens, ref_locations[0], right_answer_tokens)

            for decoy in decoys:

                _, decoy_answer_tokens = tokenize(tokenizer, decoy[0], False)
                decoy_answer_tokens_denom1 = len(sent_tokens) - 1 + len(decoy_answer_tokens) - 2
                decoy_answer_tokens_denom2 = len(nltk_sent_tokens) - len(word_tokenize(decoy[0]))
                decoy_answer_raw_score1 = base1(args, model, tokenizer, sent_tokens, ref_locations[0],
                                                decoy_answer_tokens)
                decoy_answer_raw_score2 = base2(args, model, tokenizer, sent_tokens, ref_locations[0],
                                                decoy_answer_tokens)
                decoy_answer_raw_score3 = base3(args, model, tokenizer, sent_tokens, ref_locations[0],
                                                decoy_answer_tokens)

                stats[lang]['__total__'] += 1
                if right_answer_raw_score1 > decoy_answer_raw_score1:
                    stats[lang]['__1raw__'] += 1
                if right_answer_raw_score1 / right_answer_tokens_denom1 > decoy_answer_raw_score1 / decoy_answer_tokens_denom1:
                    stats[lang]['__1tok__'] += 1
                if right_answer_raw_score1 / right_answer_tokens_denom2 > decoy_answer_raw_score1 / decoy_answer_tokens_denom2:
                    stats[lang]['__1word__'] += 1

                if right_answer_raw_score2 > decoy_answer_raw_score2:
                    stats[lang]['__2raw__'] += 1
                if right_answer_raw_score2 / right_answer_tokens_denom1 > decoy_answer_raw_score2 / decoy_answer_tokens_denom1:
                    stats[lang]['__2tok__'] += 1
                if right_answer_raw_score2 / right_answer_tokens_denom2 > decoy_answer_raw_score2 / decoy_answer_tokens_denom2:
                    stats[lang]['__2word__'] += 1

                if right_answer_raw_score3 > decoy_answer_raw_score3:
                    stats[lang]['__3raw__'] += 1
                if right_answer_raw_score3 / right_answer_tokens_denom1 > decoy_answer_raw_score3 / decoy_answer_tokens_denom1:
                    stats[lang]['__3tok__'] += 1
                if right_answer_raw_score3 / right_answer_tokens_denom2 > decoy_answer_raw_score3 / decoy_answer_tokens_denom2:
                    stats[lang]['__3word__'] += 1

                stats['all']['__total__'] += 1
                if right_answer_raw_score1 > decoy_answer_raw_score1:
                    stats['all']['__1raw__'] += 1
                if right_answer_raw_score1 / right_answer_tokens_denom1 > decoy_answer_raw_score1 / decoy_answer_tokens_denom1:
                    stats['all']['__1tok__'] += 1
                if right_answer_raw_score1 / right_answer_tokens_denom2 > decoy_answer_raw_score1 / decoy_answer_tokens_denom2:
                    stats['all']['__1word__'] += 1

                if right_answer_raw_score2 > decoy_answer_raw_score2:
                    stats['all']['__2raw__'] += 1
                if right_answer_raw_score2 / right_answer_tokens_denom1 > decoy_answer_raw_score2 / decoy_answer_tokens_denom1:
                    stats['all']['__2tok__'] += 1
                if right_answer_raw_score2 / right_answer_tokens_denom2 > decoy_answer_raw_score2 / decoy_answer_tokens_denom2:
                    stats['all']['__2word__'] += 1

                if right_answer_raw_score3 > decoy_answer_raw_score3:
                    stats['all']['__3raw__'] += 1
                if right_answer_raw_score3 / right_answer_tokens_denom1 > decoy_answer_raw_score3 / decoy_answer_tokens_denom1:
                    stats['all']['__3tok__'] += 1
                if right_answer_raw_score3 / right_answer_tokens_denom2 > decoy_answer_raw_score3 / decoy_answer_tokens_denom2:
                    stats['all']['__3word__'] += 1

    for baseline in ('1', '2', '3'):
        for normalizer in ('raw', 'tok', 'word'):
            for k in sorted(stats):
                print(
                    f"{args.input_file}\t{args.model}\t{k}\tbaseline_{baseline}\tnorm_{normalizer}\t{stats[k][f'__{baseline}{normalizer}__'] / max(1, stats[k]['__total__'])}\t{stats[k]['__total__']}")


if __name__ == "__main__":
    main()
