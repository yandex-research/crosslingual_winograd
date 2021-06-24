import os


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, candidate_a, candidate_b, ex_true=True):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for _
            candidate_a: string, correct candidate
            candidate_b: string, incorrect candidate
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b  # only used for train
        self.ex_true = ex_true
        # ex_true only matters for testing and has following string values:
        # "true" - LM has to pick this over others,
        # "false" - LM should not pick this over others
        # "other" - not known, not important, this is "other" candidate
        # "err_true" - Correct candidate but Spacy failed to find it. Automatically wrong
        # "err_false" - Incorrect candidate but Spacy failed to find it. Automatically correct


class DataProcessor(object):
    """Processor for the Wiki data set."""

    def read_dpr_format_train(self, source):
        examples = []
        lines = list(open(source, 'r'))
        for id_x, (sent, pronoun, candidates, candidate_a, _) in enumerate(
                zip(lines[0::5], lines[1::5], lines[2::5], lines[3::5], lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' ' + pronoun.strip() + ' ', " _ ", 1)
            cnd = candidates.split(",")
            cnd = (cnd[0].strip().lstrip(), cnd[1].strip().lstrip())
            candidate_a = candidate_a.strip().lstrip()
            if candidate_a.casefold() == cnd[0].casefold():
                candidate_b = cnd[1]
            else:
                candidate_b = cnd[0]
            examples.append(InputExample(guid, text_a, candidate_a, candidate_b, ex_true="true"))
        return examples

    def read_dpr_format_test(self, source):
        examples = []
        lines = list(open(source, 'r'))
        for id_x, (sent, pronoun, candidates, candidate_a, _) in enumerate(
                zip(lines[0::5], lines[1::5], lines[2::5], lines[3::5], lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' ' + pronoun.strip() + ' ', " _ ", 1)
            candidate_a = candidate_a.strip().lstrip()
            cnd = candidates.strip().split(",")
            cnd = (candidate.strip().lstrip() for candidate in cnd if
                   candidate.strip().lstrip().casefold() != candidate_a.casefold())
            examples.append(InputExample(guid, text_a, candidate_a, None, ex_true="true"))
            for candidate in cnd:
                examples.append(InputExample(guid, text_a, candidate, None, ex_true="other"))
        return examples

    def get_examples(self, data_dir, set_name):  # works for differently for train!
        """See base class."""
        file_names = {
            "train.txt": "train.txt",
            "self": "train.txt",
            "test.txt": "test.txt",
            "valid.txt": "valid.txt",
            "data_lang_en.txt": "data_lang_en.txt",
            "data_lang_jp.txt": "data_lang_jp.txt",
            "data_lang_fr.txt": "data_lang_fr.txt",
            "data_lang_zh.txt": "data_lang_zh.txt",
            "data_lang_ru.txt": "data_lang_ru.txt",
            "data_lang_pt.txt": "data_lang_pt.txt",
        }
        source = os.path.join(data_dir, file_names[set_name])

        if set_name in ["train.txt"]:
            return self.read_dpr_format_train(source)
        elif set_name in ["test.txt", "self", "valid.txt",
                          "data_lang_en.txt", "data_lang_jp.txt", "data_lang_fr.txt", "data_lang_zh.txt",
                          "data_lang_ru.txt", "data_lang_pt.txt"]:
            return self.read_dpr_format_test(source)
        else:
            print("Unknown set_name: ", set_name)
