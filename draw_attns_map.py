import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaModel

from utils import clean_sentence, tokenize, locate_reference, get_nltk_detokenizer, format_attention

MODEL_CLASSES = {
    'bert-base-multilingual-uncased': (
        BertTokenizer, BertForMaskedLM, '[MASK]', '##', ((9, 10), (10, 1), (7, 4), (9, 2), (0, 0))),
    'xlm-roberta-large': (
        XLMRobertaTokenizer, XLMRobertaModel, '<mask>', '▁', ((18, 5), (17, 7), (18, 2), (18, 10), (12, 6))),
}


def calc_vectors(A):
    A = torch.stack(A)
    v_mean = A.mean(axis=0).flatten().tolist()
    v_max, _ = torch.max(A, dim=0)
    v_max = v_max.flatten().tolist()
    return v_mean, v_max


def calc_attn_map(args, model, tokenizer, lang, pronoun_pos, sentence_a):
    inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    attention = model(input_ids.to(args.device))[-1]

    attn = format_attention(attention)
    weights = defaultdict(float)

    for layer, head in args.selected_heads:
        for pos, weight in enumerate(attn[layer, head, pronoun_pos, :].tolist()[0]):
            weights[pos] += weight

    if args.model == 'bert-base-multilingual-uncased':
        if 1:  # tokens[1] != args.token_prefix:
            merged_tokens = []
            merged_weights = []
            patched_pronoun_pos = [-1, ]
            for pos, token in enumerate(tokens):
                if '[' in token or '<' in token or not token.startswith(args.token_prefix):
                    merged_tokens.append(token)
                    merged_weights.append([weights[pos], ])
                else:
                    merged_tokens[-1] += token.strip(args.token_prefix)
                    merged_weights[-1].append(weights[pos])
                if pos == pronoun_pos[0]:
                    patched_pronoun_pos[0] = len(merged_tokens) - 1
            weights = [np.mean(w) for w in merged_weights]  # merged_weights
            tokens = merged_tokens
            pronoun_pos = patched_pronoun_pos
    else:
        if tokens[1] != args.token_prefix and lang != 'zh':
            merged_tokens = []
            merged_weights = []
            patched_pronoun_pos = [-1, ]
            for pos, token in enumerate(tokens):
                if '[' in token or '<' in token or token.startswith(args.token_prefix):
                    merged_tokens.append(token.strip(args.token_prefix))
                    merged_weights.append([weights[pos], ])
                else:
                    merged_tokens[-1] += token
                    merged_weights[-1].append(weights[pos])
                if pos == pronoun_pos[0]:
                    patched_pronoun_pos[0] = len(merged_tokens) - 1
            weights = [np.sum(w) for w in merged_weights]  # merged_weights
            tokens = merged_tokens
            pronoun_pos = patched_pronoun_pos

    res_html = [f'<!-- {sentence_a} -->', ]
    boost = 2.
    for pos, token in enumerate(tokens):
        if token == args.token_prefix:
            continue
        if token in ('[CLS]', '[SEP]', '<s>', '</s>'):
            continue
        token = token.strip(args.token_prefix)
        weight = weights[pos]
        brightness = int((255. * (len(args.selected_heads) - boost * weight)) / len(args.selected_heads))
        brightness_hex = str(hex(brightness))[2:]
        html_color = f"#ff{brightness_hex}{brightness_hex}"
        if pos == pronoun_pos[0]:
            html_color = '#c0c0c0'
        res_html.append(
            f'<span style="background-color:{html_color}">{token.replace("<", "&lt;").replace(">", "&gt;")}</span>')
    if lang in ('zh', 'jp'):
        print("".join(res_html) + "<br/>", file=args.ofh, flush=True)
    else:
        print("&nbsp;".join(res_html) + "<br/>", file=args.ofh, flush=True)


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model", default='bert-base-multilingual-uncased', # None, type=str, required=True,
    parser.add_argument("--model", default='xlm-roberta-large',  # None, type=str, required=True,
                        help="Modelname selected in the list: " + ", ".join(list(MODEL_CLASSES.keys())))
    parser.add_argument("--input_file", default='dataset.selected.tsv', type=str,  # required=True,
                        help="The input .tsv file.")
    parser.add_argument("--output_file", default='map.html', type=str,  # required=True,
                        help="The output file with attention vectors.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    assert args.model in MODEL_CLASSES, 'Unknown model!'

    tokenizer_class, model_class, mask_token, token_prefix, selected_heads = MODEL_CLASSES.get(args.model)

    args.mask_token = mask_token
    args.token_prefix = token_prefix
    args.selected_heads = selected_heads

    tokenizer = tokenizer_class.from_pretrained(args.model, do_lower_case=True)
    model = model_class.from_pretrained(args.model, output_attentions=True)
    model.to(args.device)

    with open(args.output_file, 'w', encoding='utf-8') as ofh:
        args.ofh = ofh
        print('<head><meta charset="UTF-8"></head>', file=ofh)
        with open(args.input_file, encoding='utf-8') as ifh:
            for line in ifh:
                chunks = line.strip().split('\t')
                lang = chunks[0]
                ref = json.loads(chunks[5])
                answers = json.loads(chunks[6])
                sentence = clean_sentence(chunks[3])
                en_sentence = chunks[2]

                sent_input_ids, sent_tokens = tokenize(tokenizer, sentence)

                nltk_sent_tokens = json.loads(chunks[4])
                nltk_detokenizer = get_nltk_detokenizer(nltk_sent_tokens, sentence, lang)

                # deal with reference tokens location, despite artefacts of given tokenizer
                ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence)
                if not ref_locations:
                    ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence,
                                                                 method='patch')
                if not ref_locations:
                    ref_tokens, ref_locations = locate_reference(args, tokenizer, ref[0], sent_tokens, sentence,
                                                                 method='approx')
                assert len(ref_locations), 'The reference not found in the sent_tokens.'

                calc_attn_map(args, model, tokenizer, lang, pronoun_pos=ref_locations, sentence_a=sentence)
                if en_sentence != '?':
                    print(f"<i>({en_sentence})</i><br/>", file=args.ofh, flush=True)


if __name__ == "__main__":
    main()
