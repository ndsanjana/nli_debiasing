## https://aclanthology.org/2024.eacl-long.116.pdf - Generate a dataset with external negation prefixes
import os
import json
import torch
import spacy
import random


DATASET = "MNLI"
NUM_NEG = 1  # number of times to repeat external negation prefix
MULTI_NEG = False  # True => create D^{<=NUM_NEG}; False => create D_{NT/F}^{NUM_NEG}
CREATE_DEV = False  # create dev set for inoc. (ONLY original examples---no examples in DEV have ext. neg. prefix)
CREATE_ADV = (
    False  # create challenge test set (ALL examples in ADV have ext. neg. prefix)
)
CREATE_TRAIN = (
    False  # create inoc. train set (ALL examples in TRAIN have ext. neg. prefix)
)
NEG_TYPE = "nt"  # 'nt' => "it is not true that"; 'f' => "it is false that"
FILEPATH = ""

# Create negated labeled dev sets for evaluation (matched and mismatched)
CREATE_DEV_NEG = True
DEV_SPLITS = ["matched", "mismatched"]

_LBL_DICT = {
    0: {
        "CONTRADICTION": "CONTRADICTION",
        "NEUTRAL": "NEUTRAL",
        "ENTAILMENT": "ENTAILMENT",
    },
    1: {
        "CONTRADICTION": "ENTAILMENT",
        "NEUTRAL": "NEUTRAL",
        "ENTAILMENT": "CONTRADICTION",
    },
}
_NER_MODEL = spacy.load("en_core_web_sm")


def decap_first_w(s):
    doc, min_idx = _NER_MODEL(s), 1

    if len(doc.ents) > 0:
        min_idx = min(ent.start_char for ent in doc.ents)

    if min_idx == 0 or (s[0].lower() == "i" and not s[1].isalnum()):
        return s[0].upper() + s[1:]

    return s[0].lower() + s[1:]


def get_neg_fn(fn_type):
    trigger_phrase = f'it is {"false" if fn_type == "f" else "not true"} that '

    def neg_(s, n=1):
        trigger = trigger_phrase * n

        return "I" + trigger[1:] + decap_first_w(s)

    return neg_


NEG_TYPE = NEG_TYPE.strip().lower()
assert NEG_TYPE in {"nt", "f"}
assert isinstance(NUM_NEG, int) and NUM_NEG > 0
assert CREATE_DEV or CREATE_ADV or CREATE_TRAIN or CREATE_DEV_NEG
FILEPATH = (os.path.abspath(FILEPATH) + "/").replace("//", "/")
torch.manual_seed(1)
random.seed(1)

# Output and data directories
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "mnli", "negated"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "mnli", "MNLI"
)

# Output directory for generated negated datasets
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "mnli", "negated"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DATASET == "MNLI":
    dev_file_name = "dev_matched.tsv"
    _S1_IDX, _S2_IDX, _LBL_IDX, _TREE_IDX = 8, 9, -1, 7
elif DATASET == "SNLI":
    dev_file_name = "snli_1.0_test.txt"
    _S1_IDX, _S2_IDX, _LBL_IDX, _TREE_IDX = 5, 6, 0, 4
else:
    raise NotImplementedError

neg = get_neg_fn(NEG_TYPE)
train_file_out = {k: [] for k in _LBL_DICT.keys()}
adv_file_out = {k: [] for k in _LBL_DICT.keys()}
dev_file_out = []

with open(os.path.join(DATA_DIR, dev_file_name), "r", encoding="utf-8") as f:
    dev_file = f.readlines()

# Build negated labeled dev sets (evaluation only; no train data used)
if CREATE_DEV_NEG and DATASET == "MNLI":
    suffix = f"{NUM_NEG}{NEG_TYPE}" + ("_MULTI" if MULTI_NEG else "")
    for split in DEV_SPLITS:
        with open(
            os.path.join(DATA_DIR, f"dev_{split}.tsv"), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        rows = []
        for items in map(lambda x: x.split("\t"), lines[1:]):
            s1 = items[_S1_IDX].strip()
            s2 = items[_S2_IDX].strip()
            label = items[_LBL_IDX].strip().upper()

            if label in {"", "-", "NA"}:
                continue

            if s1 and s1[-1] not in {".", "!", "?"}:
                s1 += "."

            num_neg = (
                NUM_NEG if not MULTI_NEG else NUM_NEG
            )  # apply configured negations uniformly
            new_label = _LBL_DICT[num_neg % 2][label]
            rows.append((s1, neg(s2, num_neg), new_label))

        out_path = os.path.join(
            OUTPUT_DIR, f"{DATASET.lower()}_dev_{split}_{suffix}.tsv"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("premise\thypothesis\tlabel\n")
            for p, h, y in rows:
                f.write(f"{p}\t{h}\t{y}\n")

if CREATE_DEV:
    for items in map(lambda x: x.split("\t"), dev_file[1:]):
        s1 = items[_S1_IDX].strip()
        s2 = items[_S2_IDX].strip()
        label = items[_LBL_IDX].strip().upper()

        if s1[-1] not in {".", "!", "?"}:
            s1 += "."

        dev_file_out.append({"s1": s1, "s2": s2[0].upper() + s2[1:], "label": label})

if CREATE_ADV or CREATE_TRAIN:
    with open(FILEPATH + DATASET.lower() + "_train.txt", "r", encoding="utf-8") as f:
        train_file = f.readlines()

    len_train, i = len(train_file) - 1, -1
    train_file_indices = torch.randperm(len_train).tolist()
    lbl_cnt_train = {k: 0 for k in _LBL_DICT.keys()}
    lbl_cnt_adv = {k: 0 for k in _LBL_DICT.keys()}
    lbl_lim_train = (len(dev_file) // 3) if CREATE_TRAIN else 0
    lbl_lim_adv = (len(dev_file) // 6) if CREATE_ADV else 0
    max_size, size_cnt = (lbl_lim_train + lbl_lim_adv) * 3, 0

    while size_cnt < max_size:
        i += 1
        items = train_file[train_file_indices[i] + 1].split("\t")
        label = items[_LBL_IDX].strip().upper()
        s2 = items[_S2_IDX].strip()

        if (
            label in lbl_cnt_train.keys()
            and items[_TREE_IDX].strip()[:9] == "(ROOT (S "
            and "?" not in s2
        ):
            if lbl_cnt_train[label] < lbl_lim_train:
                lbl_cnt_train[label] += 1
                out_file = train_file_out
            elif lbl_cnt_adv[label] < lbl_lim_adv:
                lbl_cnt_adv[label] += 1
                out_file = adv_file_out
            else:
                continue

            size_cnt += 1
            s1 = items[_S1_IDX].strip()
            out_file[label].append(
                {
                    "s1": s1 + ("" if s1[-1] in {".", "!", "?"} else "."),
                    "s2": s2,
                    "index": train_file_indices[i] + 1,
                }
            )

if CREATE_ADV or CREATE_TRAIN:
    for out_file in [train_file_out, adv_file_out]:
        len_file = len(out_file.get("NEUTRAL", []))
        assert sum(len(x) == len_file for _, x in out_file.items()) == 3

        if len_file > 0:  # i.e. if CREATE_[SPLIT] = True
            for lbl, lbl_list in out_file.items():
                random.shuffle(lbl_list)

                if MULTI_NEG:
                    split_size = int(len_file // NUM_NEG)
                    total_coverage = split_size * NUM_NEG
                    remainder = len_file - total_coverage
                    num_neg_fn = lambda z: z + 1
                else:
                    total_coverage, remainder, split_size = 0, 0, len_file
                    num_neg_fn = lambda z: NUM_NEG

                for i in range(NUM_NEG if MULTI_NEG else 1):
                    num_neg = num_neg_fn(i)
                    lbl_dict = _LBL_DICT[num_neg % 2]

                    for j in range(i * split_size, (i + 1) * split_size):
                        lbl_list[j].update(
                            {
                                "s2": neg(lbl_list[j]["s2"], num_neg),
                                "label": lbl_dict[lbl],
                            }
                        )

                if remainder > 0:
                    for i in range(remainder):
                        num_neg = i % NUM_NEG
                        lbl_list[total_coverage + i].update(
                            {
                                "s2": neg(lbl_list[total_coverage + i]["s2"], num_neg),
                                "label": _LBL_DICT[num_neg % 2][lbl],
                            }
                        )

    train_file_out = sum((x for _, x in train_file_out.items()), [])
    adv_file_out = sum((x for _, x in adv_file_out.items()), [])
    random.shuffle(train_file_out)
    random.shuffle(adv_file_out)
    multi = "_MULTI" if MULTI_NEG else ""

    for split, out_list in [("adv", adv_file_out), ("train", train_file_out)]:
        if len(out_list) > 0:
            out_path = os.path.join(
                OUTPUT_DIR, f"{DATASET.lower()}_{split}_{NUM_NEG}{NEG_TYPE}{multi}.json"
            )
            with open(out_path, "w") as f:  # TODO
                json.dump(out_list, f)

if len(dev_file_out) > 0:
    out_path = os.path.join(OUTPUT_DIR, f"{DATASET.lower()}_dev.json")
    with open(out_path, "w") as f:
        json.dump(dev_file_out, f)
