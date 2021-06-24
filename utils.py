import string

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer


def clean_sentence(sentence):
    return sentence.replace(' , ', ', ').replace(" 's", "'s").replace(' .', '.')


def tokenize(tokenizer, txt, spec=True):
    inputs = tokenizer.encode_plus(txt, None, return_tensors='pt', add_special_tokens=spec)
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return input_ids, tokens


def locate(sent_tok, needle_tok):
    found = []
    l = len(needle_tok)
    for idx in range(len(sent_tok) - len(needle_tok)):
        if tuple(sent_tok[idx:idx + l]) == tuple(needle_tok):
            found.append(idx)
    return found


def locate_reference(args, tokenizer, ref, sent_tokens, sentence, method='exact'):
    _, ref_tokens = tokenize(tokenizer, ref, False)
    if method == 'approx':
        if sentence.count(ref) != 1: return [], []
        if len(ref_tokens) != 1 and ref_tokens[0] == args.token_prefix:
            ref_tokens = ref_tokens[1:]
        if len(ref_tokens) and ref_tokens[0].startswith(args.token_prefix):
            ref_tokens[0] = ref_tokens[0][len(args.token_prefix):]
        if len(ref_tokens) > 1: return [], []
        for idx, tok in enumerate(sent_tokens):
            if ref_tokens[0] in tok:
                return [tok, ], [idx, ]
    if method == 'patch':
        ref = ref.rstrip('ら女.,のにを')
        if ref == '彼は彼': ref = '彼'
        if ref == '彼女を': ref = '彼'
        if ref == '彼女の': ref = '彼'
        if ref == '彼ら': ref = '彼'

        _, ref_tokens = tokenize(tokenizer, ref, False)
        if len(ref_tokens) != 1 and ref_tokens[0] == args.token_prefix:
            ref_tokens = ref_tokens[1:]
        if len(ref_tokens) and ref_tokens[0].startswith(args.token_prefix):
            ref_tokens[0] = ref_tokens[0][len(args.token_prefix):]
        if len(ref_tokens) > 1 and ref_tokens[0].startswith('彼'):
            ref_tokens = ref_tokens[:1]

    ref_locations = locate(sent_tokens, ref_tokens)
    return ref_tokens, ref_locations


def list_positions(args, tokenizer, sent_tokens, answer):
    _, a_tokens = tokenize(tokenizer, answer, False)
    a_locs = locate(sent_tokens, a_tokens)
    indices = []
    for q in a_locs:
        indices.extend([q + i for i in range(len(a_tokens))])
    if not indices and a_tokens[0].startswith(args.token_prefix):
        if len(a_tokens[0]) > len(args.token_prefix):
            a_tokens[0] = a_tokens[0][len(args.token_prefix):]
        else:
            a_tokens = a_tokens[1:]
        a_locs = locate(sent_tokens, a_tokens)
        indices = []
        for q in a_locs:
            indices.extend([q + i for i in range(len(a_tokens))])
    if not indices and not a_tokens[0].startswith(args.token_prefix):
        a_tokens[0] = args.token_prefix + a_tokens[0]
        a_locs = locate(sent_tokens, a_tokens)
        indices = []
        for q in a_locs:
            indices.extend([q + i for i in range(len(a_tokens))])
    if not indices and not a_tokens[0].startswith(args.token_prefix):
        a_tokens = [args.token_prefix] + a_tokens
        a_locs = locate(sent_tokens, a_tokens)
        indices = []
        for q in a_locs:
            indices.extend([q + i for i in range(len(a_tokens))])
    return indices


def simple_detokenizer(tokens):
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def get_nltk_detokenizer(tokens, ref, lang):
    if simple_detokenizer(tokens) == ref:
        return simple_detokenizer
    elif simple_detokenizer(tokens).replace(" n't", "n't") == ref:
        return lambda x: simple_detokenizer(x).replace(" n't", "n't")
    elif TreebankWordDetokenizer().detokenize(tokens) == ref:
        return TreebankWordDetokenizer().detokenize
    elif simple_detokenizer(tokens).replace(" ", "") == ref and lang == 'jp':
        return lambda x: simple_detokenizer(x).replace(" ", "")
    elif TreebankWordDetokenizer().detokenize(tokens).replace(" ", "") == ref.strip(' ') and lang == 'zh':
        return lambda x: TreebankWordDetokenizer().detokenize(x).replace(" ", "")
    return None


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results
