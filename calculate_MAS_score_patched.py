import argparse
import json
from collections import defaultdict

import torch
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaModel

from utils import (clean_sentence, tokenize, locate_reference, list_positions,
                   get_nltk_detokenizer,
                   format_attention, find_sub_list)

MODEL_CLASSES = {
    'bert-base-multilingual-uncased': (BertTokenizer, BertForMaskedLM, '[MASK]', '##'),
    'xlm-roberta-large': (XLMRobertaTokenizer, XLMRobertaModel, '<mask>', '▁'),
}


def MAS(args, model, tokenizer, pronoun, candidate_a, candidate_b, sentence_a, sentence_b=None, layer=None, head=None):
    """
    Computes the Maximum Attention Score (MAS) given a sentence, a pronoun and candidates for substitution.
    Parameters
    ----------
    model : transformers.BertModel
        BERT model from BERT visualization that provides access to attention
    tokenizer:  transformers.tokenization.BertTokenizer
        BERT tolenizer
    pronoun: string
        pronoun to be replaced by a candidate
    candidate_a: string
        First pronoun replacement candidate
    candidate_b: string
        Second pronoun replacement candidate
    sentence_a: string
       First, or only sentence
    sentence_b: string (optional)
        Optional, second sentence
    layer: None, int
        If none, MAS will be computed over all layers, otherwise a specific layer
    head: None, int
        If none, MAS will be compputer over all attention heads, otherwise only at specific head
    Returns
    -------
    
    activity : list
        List of scores [score for candidate_a, score for candidate_b]
    """

    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if args.model != 'xlm-roberta-large':
        token_type_ids = inputs['token_type_ids']

    candidate_a_ids = tokenizer.encode(candidate_a)[1:-1]
    candidate_b_ids = tokenizer.encode(candidate_b)[1:-1]
    pronoun_ids = tokenizer.encode(pronoun)[1:-1]

    if args.model != 'xlm-roberta-large':
        if next(model.parameters()).is_cuda:
            attention = model(input_ids.cuda(), token_type_ids=token_type_ids.cuda())[-1]
        else:
            attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    else:
        if next(model.parameters()).is_cuda:
            attention = model(input_ids.cuda())[-1]
        else:
            attention = model(input_ids)[-1]
    attn = format_attention(attention)

    if next(model.parameters()).is_cuda:
        A = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
        B = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
    else:
        A = torch.zeros((attn.shape[0], attn.shape[1]))
        B = torch.zeros((attn.shape[0], attn.shape[1]))

    if not layer is None:
        assert layer < attn.shape[0], "Maximum layer number " + str(attn.shape[0]) + " exceeded"
        layer_slice = slice(layer, layer + 1, 1)
    else:
        layer_slice = slice(None, None, None)

    if not head is None:
        assert head < attn.shape[1], "Maximum head number " + str(attn.shape[1]) + " exceeded"
        head_slice = slice(head, head + 1, 1)
    else:
        head_slice = slice(None, None, None)

    assert len(find_sub_list(pronoun_ids, input_ids[0].tolist())) > 0, "pronoun not found in sentence"
    assert len(find_sub_list(candidate_a_ids, input_ids[0].tolist())) > 0, "candidate_a not found in sentence"
    assert len(find_sub_list(candidate_b_ids, input_ids[0].tolist())) > 0, "candidate_b not found in sentence"

    for _, src in enumerate(find_sub_list(pronoun_ids, input_ids[0].tolist())):

        for _, tar_a in enumerate(find_sub_list(candidate_a_ids, input_ids[0].tolist())):
            A = A + attn[layer_slice, head_slice, slice(tar_a[0], tar_a[1] + 1, 1), slice(src[0], src[0] + 1, 1)].mean(
                axis=2).mean(axis=2)

        for _, tar_b in enumerate(find_sub_list(candidate_b_ids, input_ids[0].tolist())):
            B = B + attn[layer_slice, head_slice, slice(tar_b[0], tar_b[1] + 1, 1), slice(src[0], src[0] + 1, 1)].mean(
                axis=2).mean(axis=2)

    score = sum((A >= B).flatten()).item() / (A.shape[0] * A.shape[1])
    return [score, 1.0 - score]


def MAS_patched(args, model, tokenizer, pronoun_pos, candidate_a_pos, candidate_b_pos, sentence_a, sentence_b=None,
                layer=None, head=None, method='sum'):
    """
    Computes the Maximum Attention Score (MAS) given a sentence, a pronoun and candidates for substitution.
    Parameters
    ----------
    model : transformers.BertModel
        BERT model from BERT visualization that provides access to attention
    tokenizer:  transformers.tokenization.BertTokenizer
        BERT tolenizer
    pronoun: string
        pronoun to be replaced by a candidate
    candidate_a: string
        First pronoun replacement candidate
    candidate_b: string
        Second pronoun replacement candidate
    sentence_a: string
       First, or only sentence
    sentence_b: string (optional)
        Optional, second sentence
    layer: None, int
        If none, MAS will be computed over all layers, otherwise a specific layer
    head: None, int
        If none, MAS will be compputer over all attention heads, otherwise only at specific head
    Returns
    -------
    
    activity : list
        List of scores [score for candidate_a, score for candidate_b]
    """

    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if args.model != 'xlm-roberta-large':
        token_type_ids = inputs['token_type_ids']

    if args.model != 'xlm-roberta-large':
        if next(model.parameters()).is_cuda:
            attention = model(input_ids.cuda(), token_type_ids=token_type_ids.cuda())[-1]
        else:
            attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    else:
        if next(model.parameters()).is_cuda:
            attention = model(input_ids.cuda())[-1]
        else:
            attention = model(input_ids)[-1]
    attn = format_attention(attention)

    if next(model.parameters()).is_cuda:
        A = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
        B = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
    else:
        A = torch.zeros((attn.shape[0], attn.shape[1]))
        B = torch.zeros((attn.shape[0], attn.shape[1]))

    if not layer is None:
        assert layer < attn.shape[0], "Maximum layer number " + str(attn.shape[0]) + " exceeded"
        layer_slice = slice(layer, layer + 1, 1)
    else:
        layer_slice = slice(None, None, None)

    if not head is None:
        assert head < attn.shape[1], "Maximum head number " + str(attn.shape[1]) + " exceeded"
        head_slice = slice(head, head + 1, 1)
    else:
        head_slice = slice(None, None, None)

    As = []
    Bs = []
    for src in pronoun_pos:
        for tar_a in candidate_a_pos:
            As.append(attn[layer_slice, head_slice, tar_a, src])
        for tar_b in candidate_b_pos:
            Bs.append(attn[layer_slice, head_slice, tar_b, src])

    As = torch.stack(As)
    Bs = torch.stack(Bs)
    if method == 'mean':
        A = As.mean(axis=0)
        B = Bs.mean(axis=0)
    elif method == 'max':
        A, _ = torch.max(As, dim=0)
        B, _ = torch.max(Bs, dim=0)
    else:
        A = As.sum(axis=0)
        B = Bs.sum(axis=0)
    score = sum((A >= B).flatten()).item() / (A.shape[0] * A.shape[1])
    return [score, 1.0 - score]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default='xlm-roberta-large',
                        help="Modelname selected in the list: " + ", ".join(list(MODEL_CLASSES.keys())))
    parser.add_argument("--input_file", default='dataset.v4.tsv', type=str,  # required=True,
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

    stats = defaultdict(lambda: defaultdict(int))
    stats_patched = defaultdict(lambda: defaultdict(int))

    with open(args.input_file, encoding='utf-8') as ifh:
        for line in ifh:
            chunks = line.strip().split('\t')

            lang = chunks[0]
            ref = json.loads(chunks[5])
            answers = json.loads(chunks[6])
            sentence_orig = chunks[3]
            sentence = clean_sentence(chunks[3])

            sent_input_ids, sent_tokens = tokenize(tokenizer, sentence)

            nltk_sent_tokens = json.loads(chunks[4])
            nltk_detokenizer = get_nltk_detokenizer(nltk_sent_tokens, sentence, chunks[0])

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

            right_answer_positions = list_positions(args, tokenizer, sent_tokens, right_answer[0])

            for decoy in decoys:
                try:
                    sc = MAS(args, model, tokenizer, pronoun=ref[0], candidate_a=right_answer[0], candidate_b=decoy[0],
                             sentence_a=sentence_orig)
                    if sc[0] > sc[1]:
                        stats[lang][1] += 1
                        stats['all'][1] += 1
                    stats[lang]['__total__'] += 1
                    stats['all']['__total__'] += 1
                except:
                    stats[lang]['__skipped__'] += 1
                    stats['all']['__skipped__'] += 1
                    pass

                try:
                    decoy_answer_positions = list_positions(args, tokenizer, sent_tokens, decoy[0])
                    sc = MAS_patched(args, model, tokenizer, pronoun_pos=ref_locations,
                                     candidate_a_pos=right_answer_positions, candidate_b_pos=decoy_answer_positions,
                                     sentence_a=sentence, method='max')
                    if sc[0] > sc[1]:
                        stats_patched[lang][1] += 1
                        stats_patched['all'][1] += 1
                    stats_patched[lang]['__total__'] += 1
                    stats_patched['all']['__total__'] += 1
                except:
                    stats_patched[lang]['__skipped__'] += 1
                    stats_patched['all']['__skipped__'] += 1
                    print(chunks[:5])
                    pass

    for k in sorted(stats):
        print(
            f"{args.input_file}\t{args.model}\t{k}\tMAS\t{stats[k][1] / max(1, stats[k]['__total__'])}"
            f"\t{stats[k]['__total__']}\t{stats[k]['__skipped__']}")

    stats = stats_patched
    for k in sorted(stats):
        print(
            f"{args.input_file}\t{args.model}\t{k}\tMAS_patched\t{stats[k][1] / max(1, stats[k]['__total__'])}\t"
            f"{stats[k]['__total__']}\t{stats[k]['__skipped__']}")


if __name__ == "__main__":
    main()
