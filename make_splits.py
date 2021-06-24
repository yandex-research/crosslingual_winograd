import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split


def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def get_examples(filename, use_lang, split=0., seed=1):
    texts = []
    with open(filename, encoding='utf-8') as ifh:
        for line in ifh:
            chunks = line.strip().split('\t')
            lang = chunks[0]
            if use_lang != lang:
                continue
            texts.append(line.strip())

    if split == 0.:
        return texts, []
    X_train, X_test = train_test_split(texts, random_state=seed, train_size=split)
    return X_train, X_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='dataset.tsv', type=str,
                        help="The input .tsv file.")
    parser.add_argument("--output_dir", default='splits', type=str,
                        help="The output folder.")
    parser.add_argument("--folds", default=5, type=int,
                        help="The number of folds.")
    args = parser.parse_args()

    my_seed = 0
    set_seed(my_seed)

    for _l in ('en', 'ru', 'pt', 'jp', 'zh', 'fr'):
        base_examples, test_examples = get_examples(args.input_file, _l, split=0.9, seed=my_seed)
        print(f'LANG {_l}')
        print(f'base {len(base_examples)}')
        print(f'test {len(test_examples)}')

        with open(f'{args.output_dir}/lang_{_l}_test.tsv', 'w', encoding='utf-8') as fh:
            print('\n'.join(test_examples), file=fh)

        for subseed in range(args.folds):
            train, dev = train_test_split(base_examples, random_state=subseed, test_size=len(test_examples))
            print(f'train{subseed}  {len(train)}   dev{subseed}  {len(dev)}')
            with open(f'{args.output_dir}/lang_{_l}_fold{subseed}_train.tsv', 'w', encoding='utf-8') as fh:
                print('\n'.join(train), file=fh)
            with open(f'{args.output_dir}/lang_{_l}_fold{subseed}_dev.tsv', 'w', encoding='utf-8') as fh:
                print('\n'.join(dev), file=fh)


if __name__ == "__main__":
    main()
