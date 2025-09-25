import datasets
import os, sys, re
import numpy as np
import argparse
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
# from nlpaug.util import Action

# Download WordNet
import nltk

NLTK_RESOURCES = [
    ("corpora", "wordnet"),
    ("taggers", "averaged_perceptron_tagger"),
    ("taggers", "averaged_perceptron_tagger_eng"),
]


def _ensure_nltk_resources():
    for path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(f"{path}/{name}")
        except LookupError:
            try:
                nltk.download(name)
            except Exception:
                pass


_ensure_nltk_resources()

parser = argparse.ArgumentParser(
    description="NLI data augmentation with nlpaug...)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--input",
    required=True,
    help="data input file (json/jsonl/csv/tsv), readable by hugging face dataset method",
)
parser.add_argument(
    "--task",
    type=str,
    choices=["char", "emb", "synonym", "tfidf", "wordnet"],
    required=True,
    help="type of data augmentation: ['char', 'emb', 'synonym', 'tfidf', 'wordnet']",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model path for data augmentation, require if if task is 'emb' or 'synonym', please refer to https://github.com/makcedward/nlpaug/tree/master for more detail",
)
parser.add_argument(
    "--out", type=str, default="out.json", help="output argumentated data"
)


def do_aug(sen, agu, trial=20):
    for i in range(trial):
        new_sen = agu.augment(sen)[0]
        new_sen.replace("_", " ")
        if new_sen != sen:
            break
    return new_sen


def process_dataset(data, aug):
    new_data = []
    for ex in data:
        premise = ex["premise"]
        sen = ex["hypothesis"]
        label = ex["label"]
        hypothesis = do_aug(sen, aug)
        tem_dict = {"premise": premise, "hypothesis": hypothesis, "label": label}
        new_data.append(tem_dict)
    return new_data


def write_json(data, fn):
    import json

    with open(fn, "w") as json_file:
        for d in data:
            if d["label"] > -1:
                json.dump(d, json_file)
                json_file.write("\n")


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _standardize_dataset_columns(ds):
    if "premise" not in ds.column_names and "sentence1" in ds.column_names:
        ds = ds.rename_column("sentence1", "premise")
    if "hypothesis" not in ds.column_names and "sentence2" in ds.column_names:
        ds = ds.rename_column("sentence2", "hypothesis")
    if "label" not in ds.column_names:
        for candidate in ["gold_label", "label1"]:
            if candidate in ds.column_names:
                ds = ds.rename_column(candidate, "label")
                break
    return ds


def _convert_labels(example):
    label = example.get("label")
    if isinstance(label, str):
        label = LABEL_MAP.get(label.strip(), -1)
    elif label is None:
        label = -1
    example["label"] = label
    return example


def load_input_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".jsonl"]:
        ds = datasets.load_dataset("json", data_files={"train": path})["train"]
    elif ext in [".csv", ".tsv"]:
        delimiter = "\t" if ext == ".tsv" else ","
        ds = datasets.load_dataset(
            "csv", data_files={"train": path}, delimiter=delimiter
        )["train"]
    else:
        raise ValueError(f"Unsupported input file type: {ext}")

    ds = _standardize_dataset_columns(ds)
    if "label" in ds.column_names:
        ds = ds.map(_convert_labels)
    return ds


if __name__ == "__main__":
    args = parser.parse_args()
    input = args.input
    task = args.task
    model = args.model
    out = args.out

    # input = 'val_no_premise.json'
    # task = 'emb'
    # model = 'aug_model/GoogleNews-vectors-negative300.bin'
    # # task = 'synonym'
    # # model = 'aug_model/ppdb-2.0-m-all'
    # out = 'test.json'

    data = load_input_dataset(input)

    if task == "char":
        aug = nac.RandomCharAug(action="substitute")
        print("char augmentation: substitution!")
    elif task == "emb":
        aug = naw.WordEmbsAug(
            model_type="word2vec", model_path=model, action="substitute"
        )
        print(f"word augmentation: substitution using word2vec embeding: {model}")
        # word2vec model download:
        # from nlpaug.util.file.download import DownloadUtil
        # DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
        # DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
        # DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.')
    elif task == "synonym":
        aug = naw.SynonymAug(aug_src="ppdb", model_path=model)
        print(f"word augmentation: synonym substitution using PPDB: {model}")
        # ppdb model download: http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-m-all.gz
    elif task == "tfidf":
        import nlpaug.model.word_stats as nmw

        train_x = [x["hypothesis"] for x in data]
        train_x_tokens = [_tokenizer(x) for x in train_x]
        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save(".")
        aug = naw.TfIdfAug(model_path=".", tokenizer=_tokenizer)
        print(f"word augmentation: substitution using tfidf")
    elif task == "wordnet":
        aug = naw.SynonymAug(aug_src="wordnet")
        print(f"word augmentation: synonym substitution using wordnet")

    new_data = process_dataset(data, aug)
    write_json(new_data, out)


# python do_syn_augmentation.py --input E:\Code\Gemma Finetuning for NLI\data\mnli\MNLI\dev_matched.tsv --out E:\Code\Gemma Finetuning for NLI\data\Augmented\synonym\aug_val_data.json --task wordnet --model None
