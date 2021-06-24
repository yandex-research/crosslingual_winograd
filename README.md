# It’s All in the Heads

This repository contains supplementary materials for the [paper](https://arxiv.org/abs/2106.12066) 
**"It’s All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning"**
(Findings of ACL 2021).

___

## Installation

To run the experiments, you will need the Hugging Face [transformers](https://github.com/huggingface/transformers)
library, any version starting from 4.0.1 should work. Also, the code uses `nltk`, `torch`, `numpy`, and `scikit-learn`.

For specific dependency versions, you can use [requirements.txt](./requirements.txt)

To run the supervised finetuning baselines, you will also need to install [Apex](https://github.com/NVIDIA/apex).

## How can I run it?

* Use the input file dataset.tsv provided.
* To compute the MAS baseline, run:

```
python calculate_MAS_score_patched.py --model bert-base-multilingual-uncased --input_file dataset.tsv > results.mbert.MAS.txt
python calculate_MAS_score_patched.py --model xlm-roberta-large --input_file dataset.tsv > results.xlmr.MAS.txt
```

* To compute several unsupervised baselines, run:

```
python calculate_baselines.py --model xlm-roberta-large --input_file dataset.tsv > results.xlmr.baselines.txt
python calculate_baselines.py --model bert-base-multilingual-uncased --input_file dataset.tsv > results.mbert.baselines.txt
```

* To run the supervised baselines, proceed to the [supervised_baselines](supervised_baselines) directory and run the
  following scripts:

For multilingual BERT: `bash run_mbert.sh`

For XLM-R: `bash run_xlm_roberta.sh`

These baselines use code from the [bert-commonsense](https://github.com/vid-koci/bert-commonsense) repository for
the [paper](https://www.aclweb.org/anthology/P19-1478/) "A Surprisingly Robust Trick for Winograd Schema Challenge"
(Kocijan et al., 2019).

* Finally, to calculate scores of the proposed method using multilingual BERT, run:

```
python dump_attns.py --model bert-base-multilingual-uncased --input_file dataset.tsv --output_file dump.mbert.attn.tsv
mkdir splits_mbert
python make_splits.py --input_file dump.mbert.attn.tsv --output_dir splits_mbert
python calculate_scores_on_splits.py splits_mbert result_scores.mbert.tsv
```

* Same, with the XLM-Roberta model:

```
python dump_attns.py --model xlm-roberta-large --input_file dataset.tsv --output_file dump.xlmr.attn.tsv
mkdir splits_xlmr
python make_splits.py --input_file dump.xlmr.attn.tsv --output_dir splits_xlmr
python calculate_scores_on_splits.py splits_xlmr result_scores.xlmr.tsv
```

* To draw the attention visualization, use:

```
python draw_attns_map.py --model xlm-roberta-large --output_file map.html --input_file dataset.selected.tsv
```

## XWINO

XWINO is a multilingual collection of Winograd Schemas in six languages that can be used for evaluation of cross-lingual
commonsense reasoning capabilities. 

The datasets that comprise XWINO are:
1) The original [Winograd Schema Challenge](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSCollection.xml) ([Levesque](http://www.cs.toronto.edu/~hector/Papers/winograd.pdf), 2012);
2) Additional data from the [SuperGLUE](https://super.gluebenchmark.com/tasks/) WSC benchmark ([Wang et al.](https://papers.nips.cc/paper/2019/hash/4496bf24afe7fab6f046bf4923da8de6-Abstract.html), 2019);
3) The [Definite Pronoun Resolution](http://www.hlt.utdallas.edu/~vince/data/emnlp12/) dataset ([Rahman and Ng](https://www.aclweb.org/anthology/D12-1071/), 2012) (accessed from https://github.com/Yre/wsc_naive);
2) A collection of [French Winograd Schemas](http://www.llf.cnrs.fr/fr/winograd-fr) ([Amsili and Seminck](https://www.aclweb.org/anthology/W17-1504/), 2017);
3) [Japanese translation](https://github.com/ku-nlp/Winograd-Schema-Challenge-Ja) of Winograd Schema Challenge ([柴田知秀 et al.](http://www.anlp.jp/proceedings/annual_meeting/2015/pdf_dir/E3-1.pdf), 2015);
4) [Russian Winograd Schema Challenge](https://russiansuperglue.com/tasks/task_info/RWSD) ([Shavrina et al.](https://www.aclweb.org/anthology/2020.emnlp-main.381/), 2020);
5) A collection of [Winograd Schemas in Chinese](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSChinese.html);
6) Winograd Schemas [in Portuguese](https://github.com/gabimelo/portuguese_wsc) ([Melo et al.](https://www.teses.usp.br/teses/disponiveis/3/3141/tde-14012021-124730/es.php), 2019).

The columns of the TSV-formatted [dataset](./dataset.tsv) are:

1) A two-letter language code (ISO 639-1);
2) Source dataset identifier
3) English reference schema (if exists, else "?")
4) Schema raw text 
5) JSON of NLTK-tokenized sentence text
6) JSON with the reference pronoun specification: `[<raw text>, <token ids range>, <tokenized>]`
7) JSON with answer candidates specification: `[[<answer1 raw text>, <answer1 token ids range>, <answer1 tokenized>, <correct answer (binary)>], [<answer2 raw text>, <answer2 token ids range>, <answer2 tokenized>, <correct answer (binary)>]]`

## References

```
@misc{tikhonov2021heads,
    title={It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
    author={Alexey Tikhonov and Max Ryabinin},
    year={2021},
    eprint={2106.12066},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```