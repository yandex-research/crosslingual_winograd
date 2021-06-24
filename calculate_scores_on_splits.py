import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

FOLDS = 5

LANG_LIST = ['en', 'fr', 'jp', 'ru', 'zh', 'pt']

features_shift = {
    ('mean', 'MAS'): -8,
    ('max', 'MAS'): -6,
    ('mean', 'normal'): -4,
    ('max', 'normal'): -2,
}

assert len(sys.argv) > 1, 'input dir not found'
assert len(sys.argv) > 2, 'output filename not found'

input_dir = sys.argv[1]
output_filename = sys.argv[2]


def load_file(fn, normalizer, pooling, direction, use_heads):
    resX = []
    resY = []
    mas = []
    for idx, line in enumerate(open(fn, encoding='utf-8')):
        chunks = line.strip().split('\t')
        lang = chunks[0]
        right_answer, decoy = chunks[features_shift[(pooling, direction)]:len(chunks) + features_shift[
            (pooling, direction)] + 2]
        right_answer = json.loads(right_answer)
        decoy = json.loads(decoy)
        if use_heads:
            right_answer = [right_answer[idx] for idx in use_heads]
            decoy = [decoy[idx] for idx in use_heads]
        right_answer = np.array(right_answer)
        decoy = np.array(decoy)
        right_score = sum([idx for idx in right_answer])
        decoy_score = sum([idx for idx in decoy])

        if right_score > decoy_score:
            mas.append(1)
        else:
            mas.append(0)

        if normalizer == 'ident':
            item1 = right_answer
            item2 = decoy
        else:
            item1 = right_answer - decoy
            item2 = decoy - right_answer

        if idx % 2:
            resX.append(item1)
            resY.append(1)
        else:
            resX.append(item2)
            resY.append(0)

    return resX, resY, mas


def load_split(lang, split, normalizer, pooling, direction, use_heads):
    trainX, trainY, m1 = load_file(f'{input_dir}/lang_{lang}_fold{split}_train.tsv', normalizer, pooling, direction,
                                   use_heads)
    devX, devY, m2 = load_file(f'{input_dir}/lang_{lang}_fold{split}_dev.tsv', normalizer, pooling, direction,
                               use_heads)
    testX, testY, m3 = load_file(f'{input_dir}/lang_{lang}_test.tsv', normalizer, pooling, direction, use_heads)

    return trainX, trainY, m1, devX, devY, m2, testX, testY, m3


def load_sets(lang, split, normalizer, pooling, direction, use_heads):
    trainX, trainY, m1, devX, devY, m2, testX, testY, m3 = load_split(lang, split, normalizer, pooling, direction,
                                                                      use_heads)
    X_valids = defaultdict(list)
    y_valids = defaultdict(list)
    M_valids = defaultdict(list)
    for lang in LANG_LIST:
        X1, Y1, M1, X2, Y2, M2, X3, Y3, M3 = load_split(lang, '0', normalizer, pooling, direction, use_heads)
        X_valids[lang] = X1 + X2 + X3
        y_valids[lang] = Y1 + Y2 + Y3
        M_valids[lang] = M1 + M2 + M3
    return trainX, trainY, devX, devY, testX, testY, X_valids, y_valids, M_valids


# Get number of heads and fix the list of random heads
devX, _, _ = load_file(f'{input_dir}/lang_zh_fold0_dev.tsv', 'ident', 'max', 'normal', use_heads=None)
HEADS_NUM = len(devX[0])
random_heads = [np.random.choice(range(HEADS_NUM), 5).tolist() for _ in range(5)]

headers = ['', 'base lang', 'train acc', '', 'valid acc', '']
for lang in LANG_LIST:
    headers.extend([lang, ''])

with open(output_filename, 'w', encoding='utf-8') as output_file:
    for penalty, solver in (('l2', 'lbfgs'), ('l1', 'liblinear')):
        # for penalty, solver in (('l2','lbfgs'), ):
        for normalizer in ('ident', 'diff'):
            for pooling in ('mean', 'max'):
                for direction in ('MAS', 'normal'):

                    lists_of_best_heads = []

                    for train_lang in LANG_LIST:
                        best_heads = None
                        use_heads = None
                        for heads_number in (0, 1, 2, 4, 8, 16, 32):
                            if heads_number:
                                use_heads = best_heads[:heads_number]

                            score_test = []
                            score_train = []
                            score_dev = []
                            score_valids = defaultdict(list)

                            if not best_heads:
                                weights = defaultdict(float)

                            for seed in range(FOLDS):
                                X_train, y_train, X_dev, y_dev, X_test, y_test, X_valids, y_valids, M_valids = load_sets(
                                    train_lang, seed, normalizer, pooling, direction, use_heads)

                                clf = LogisticRegression(random_state=0, penalty=penalty, solver=solver).fit(X_train,
                                                                                                             y_train)

                                score_train.append(clf.score(X_train, y_train))
                                score_test.append(clf.score(X_test, y_test))
                                score_dev.append(clf.score(X_dev, y_dev))

                                for lang in LANG_LIST:
                                    score_valids[lang].append(
                                        clf.score(np.array(X_valids[lang]), np.array(y_valids[lang])))
                                if not best_heads:
                                    for q, w in enumerate(clf.coef_[0]):
                                        weights[q] += w

                            if not best_heads:
                                best_heads = [k for k, w in list(sorted(weights.items(), key=lambda x: -abs(x[1])))[:32]
                                              if abs(w) > 0]
                                lists_of_best_heads.append(best_heads)
                            res = [np.mean(score_train), np.std(score_train), np.mean(score_dev), np.std(score_dev),
                                   np.mean(score_test), np.std(score_test)]
                            n1, n2 = 0, 0
                            for lang in LANG_LIST:
                                if lang == train_lang:
                                    res.append('-')
                                    res.append('-')
                                else:
                                    res.append(np.mean(score_valids[lang]))
                                    res.append(np.std(score_valids[lang]))
                                    n1 += score_valids[lang][0] * len(y_valids[lang])
                                    n2 += len(y_valids[lang])

                            res.append(n1 / n2)

                            dump = [penalty, normalizer, pooling, direction, train_lang]
                            if not use_heads:
                                dump.append('all')
                                dump.append('-')
                            else:
                                suffix = "s" if heads_number > 1 else ""
                                dump.append(f'top{heads_number}')
                                dump.append(json.dumps(use_heads))
                            dump.append('\t'.join(map(str, res)).replace('.', ','))
                            print(f'\t'.join(map(str, dump)), file=output_file)

                    head_score = defaultdict(int)
                    for run in lists_of_best_heads:
                        for idx, i in enumerate(run[::-1]):
                            head_score[i] += idx
                    selected_heads = []
                    for head, _ in sorted(head_score.items(), key=lambda x: -x[1]):
                        selected_heads.append(head)

                    selected_heads = selected_heads[:5]

                    for train_lang in LANG_LIST:

                        score_test = []
                        score_dev = []
                        score_train = []
                        score_valids = defaultdict(list)

                        for seed in range(FOLDS):

                            X_train, y_train, X_dev, y_dev, X_test, y_test, X_valids, y_valids, M_valids = load_sets(
                                train_lang, seed, normalizer, pooling, direction, selected_heads)

                            clf = LogisticRegression(random_state=0, penalty=penalty, solver=solver).fit(X_train,
                                                                                                         y_train)
                            score_train.append(clf.score(X_train, y_train))
                            score_test.append(clf.score(X_test, y_test))
                            score_dev.append(clf.score(X_dev, y_dev))

                            for lang in LANG_LIST:
                                score_valids[lang].append(clf.score(np.array(X_valids[lang]), np.array(y_valids[lang])))

                        res = [np.mean(score_train), np.std(score_train), np.mean(score_dev), np.std(score_dev),
                               np.mean(score_test), np.std(score_test)]
                        n1, n2 = 0, 0
                        for lang in LANG_LIST:
                            if lang == train_lang:
                                res.append('-')
                                res.append('-')
                            else:
                                res.append(np.mean(score_valids[lang]))
                                res.append(np.std(score_valids[lang]))
                                n1 += score_valids[lang][0] * len(y_valids[lang])
                                n2 += len(y_valids[lang])
                        res.append(n1 / n2)

                        dump = [penalty, normalizer, pooling, direction, train_lang, 'best5',
                                json.dumps(selected_heads), '\t'.join(map(str, res)).replace('.', ',')]
                        print(f'\t'.join(map(str, dump)), file=output_file)

                        for rseed in range(len(random_heads)):
                            for seed in range(FOLDS):

                                X_train, y_train, X_dev, y_dev, X_test, y_test, X_valids, y_valids, M_valids = load_sets(
                                    train_lang, seed, normalizer, pooling, direction, random_heads[rseed])

                                clf = LogisticRegression(random_state=0, penalty=penalty, solver=solver).fit(X_train,
                                                                                                             y_train)
                                score_train.append(clf.score(X_train, y_train))
                                score_test.append(clf.score(X_test, y_test))
                                score_dev.append(clf.score(X_dev, y_dev))

                                for lang in LANG_LIST:
                                    score_valids[lang].append(
                                        clf.score(np.array(X_valids[lang]), np.array(y_valids[lang])))

                            res = [np.mean(score_train), np.std(score_train), np.mean(score_dev), np.std(score_dev),
                                   np.mean(score_test), np.std(score_test)]
                            n1, n2 = 0, 0
                            for lang in LANG_LIST:
                                if lang == train_lang:
                                    res.append('-')
                                    res.append('-')
                                else:
                                    res.append(np.mean(score_valids[lang]))
                                    res.append(np.std(score_valids[lang]))
                                    n1 += score_valids[lang][0] * len(y_valids[lang])
                                    n2 += len(y_valids[lang])
                            res.append(n1 / n2)

                            dump = [penalty, normalizer, pooling, direction, 'logreg', f'random5-{seed}',
                                    json.dumps(random_heads[rseed]), '\t'.join(map(str, res)).replace('.', ',')]
                            print(f'\t'.join(map(str, dump)), file=output_file)

                    _, _, _, _, _, _, X_valids, y_valids, M_valids = load_sets('en', '0', 'ident', pooling, direction,
                                                                               selected_heads)

                    n1, n2 = 0, 0
                    res = ['-', '-', '-', '-', '-', '-']
                    for lang in LANG_LIST:
                        res.append(np.mean(M_valids[lang]))
                        res.append('-')
                        n1 += sum(M_valids[lang])
                        n2 += len(M_valids[lang])

                    if n2:
                        res.append(n1 / n2)
                    else:
                        res.append('nan')

                    dump = [penalty, normalizer, pooling, direction, 'MAS-like', 'best5', json.dumps(selected_heads),
                            '\t'.join(map(str, res)).replace('.', ',')]
                    print(f'\t'.join(map(str, dump)), file=output_file)

                    for seed in range(len(random_heads)):
                        _, _, _, _, _, _, X_valids, y_valids, M_valids = load_sets('en', '0', 'ident', pooling,
                                                                                   direction, random_heads[seed])

                        n1, n2 = 0, 0
                        res = ['-', '-', '-', '-', '-', '-']
                        for lang in LANG_LIST:
                            res.append(np.mean(M_valids[lang]))
                            res.append('-')
                            n1 += sum(M_valids[lang])
                            n2 += len(M_valids[lang])

                        if n2:
                            res.append(n1 / n2)
                        else:
                            res.append('nan')

                        dump = [penalty, normalizer, pooling, direction, 'MAS-like', f'random5-{seed}',
                                json.dumps(random_heads[seed]), '\t'.join(map(str, res)).replace('.', ',')]
                        print(f'\t'.join(map(str, dump)), file=output_file)
