import argparse
import json

import torch
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaModel

from utils import clean_sentence, tokenize, locate_reference, list_positions, get_nltk_detokenizer, format_attention

MODEL_CLASSES = {
    'bert-base-multilingual-uncased': (BertTokenizer, BertForMaskedLM, '[MASK]', '##'),
    'xlm-roberta-large': (XLMRobertaTokenizer, XLMRobertaModel, '<mask>', '▁'),
}


def calc_vectors(A):
    A = torch.stack(A)
    v_mean = A.mean(axis=0).flatten().tolist()
    v_max, _ = torch.max(A, dim=0)
    v_max = v_max.flatten().tolist()
    return v_mean, v_max


def get_vectors(args, model, tokenizer, pronoun_pos, candidate_a_pos, candidate_b_pos, sentence_a):
    inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']

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

    layer_slice = slice(None, None, None)
    head_slice = slice(None, None, None)

    As = []
    Bs = []
    As_rev = []
    Bs_rev = []
    for src in pronoun_pos:
        for tar_a in candidate_a_pos:
            As.append(attn[layer_slice, head_slice, tar_a, src])
            As_rev.append(attn[layer_slice, head_slice, src, tar_a])
        for tar_b in candidate_b_pos:
            Bs.append(attn[layer_slice, head_slice, tar_b, src])
            Bs_rev.append(attn[layer_slice, head_slice, src, tar_b])

    r_mean, r_max = calc_vectors(As)
    d_mean, d_max = calc_vectors(Bs)
    r_mean_rev, r_max_rev = calc_vectors(As_rev)
    d_mean_rev, d_max_rev = calc_vectors(Bs_rev)

    return r_mean, d_mean, r_max, d_max, r_mean_rev, d_mean_rev, r_max_rev, d_max_rev


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default='xlm-roberta-large',
                        help="Modelname selected in the list: " + ", ".join(list(MODEL_CLASSES.keys())))
    parser.add_argument("--input_file", default='dataset.nltk_tokenized.pt.tsv', type=str,  # required=True,
                        help="The input .tsv file.")
    parser.add_argument("--output_file", default='attn_vectors.tsv', type=str,  # required=True,
                        help="The output file with attention vectors.")
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

    with open(args.output_file, 'w', encoding='utf-8') as ofh:
        with open(args.input_file, encoding='utf-8') as ifh:
            for line in ifh:
                chunks = line.strip().split('\t')
                lang = chunks[0]
                ref = json.loads(chunks[5])
                answers = json.loads(chunks[6])
                sentence = clean_sentence(chunks[3])

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

                decoys = []
                for answer in answers:
                    if answer[-1]:
                        right_answer = answer
                    else:
                        decoys.append(answer)
                right_answer_positions = list_positions(args, tokenizer, sent_tokens, right_answer[0])

                decoys_res = []
                for decoy in decoys:
                    decoy_answer_positions = list_positions(args, tokenizer, sent_tokens, decoy[0])

                    vectors = get_vectors(args, model, tokenizer, pronoun_pos=ref_locations,
                                          candidate_a_pos=right_answer_positions,
                                          candidate_b_pos=decoy_answer_positions, sentence_a=sentence)
                    print("\t".join(chunks[:5] + chunks[6:11] + list(map(json.dumps, vectors))), file=ofh)


if __name__ == "__main__":
    main()
