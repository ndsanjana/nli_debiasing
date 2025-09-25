from __future__ import annotations

import re

from model_prediction import ModelPrediction
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
import pandas as pd

# https://github.com/datngu/nli-artifacts/blob/main/aug/do_augmentation.py

MODEL_BASE_ID = "ndsanjana/BERT-Base-MNLI-Orig"
MODEL_DEBIASED_ID = "ndsanjana/BERT-Base-MNLI-Debiased"
SYNONYM_DATASETS = {
    "Synonym Augmented Val": "data/Augmented/synonym/aug_val_data.json",
    "Synonym Augmented Val Mismatched": "data/Augmented/synonym/aug_val_data_mis.json",
}


label2idx = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
idx2label = {v: k for k, v in label2idx.items()}
pred2idx = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

OVERLAP_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
OVERLAP_BIN_LABELS = ["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]


def _load_synonym_dataset(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df.rename(
        columns={
            "premise": "sentence1",
            "hypothesis": "sentence2",
            "label": "gold_label",
        }
    )

    df["gold_label"] = df["gold_label"].map(idx2label).str.upper()
    df = df.dropna(subset=["sentence1", "sentence2", "gold_label"]).reset_index(
        drop=True
    )

    unexpected = sorted(set(df["gold_label"]) - set(label2idx))
    if unexpected:
        raise ValueError("Unexpected gold labels encountered: " + ", ".join(unexpected))
    return df


def _predictions_to_idx(preds: list[str]) -> list[int]:
    mapped: list[int] = []
    for p in preds:
        if not isinstance(p, str):
            raise ValueError(f"Prediction must be string-like, received {type(p)}")
        p_up = p.strip().upper()
        if p_up in label2idx:
            mapped.append(label2idx[p_up])
        elif p_up in pred2idx:
            mapped.append(pred2idx[p_up])
        elif p_up.startswith("LABEL_"):
            try:
                mapped.append(int(p_up.split("_")[-1]))
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unable to parse predicted label: {p}") from exc
        else:
            raise ValueError(f"Unknown predicted label encountered: {p}")
    return mapped


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _compute_lexical_overlap(premise: str, hypothesis: str) -> float:
    premise_tokens = _tokenize_for_overlap(premise)
    hypothesis_tokens = _tokenize_for_overlap(hypothesis)
    if not hypothesis_tokens:
        return 0.0

    overlap_count = len(premise_tokens.intersection(hypothesis_tokens))
    overlap_ratio = overlap_count / len(hypothesis_tokens)
    return max(0.0, min(1.0, overlap_ratio))


def _evaluate_predictions(
    df: pd.DataFrame, mp: ModelPrediction
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_base: list[str] = []
    preds_debiased: list[str] = []
    lexical_overlaps: list[float] = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Synonym Augmented",
        unit="ex",
    ):
        input_base, input_debiased = mp.tokenize_input(row.sentence1, row.sentence2)
        pred_base, pred_debiased, *_ = mp.predict(input_base, input_debiased)
        preds_base.append(pred_base)
        preds_debiased.append(pred_debiased)
        lexical_overlaps.append(_compute_lexical_overlap(row.sentence1, row.sentence2))

    gold_idx = [label2idx[y] for y in df["gold_label"].tolist()]
    base_idx = _predictions_to_idx(preds_base)
    deb_idx = _predictions_to_idx(preds_debiased)

    records = []
    for model_name, pred_idx in (
        ("base", base_idx),
        ("debiased", deb_idx),
    ):
        records.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(gold_idx, pred_idx),
                "precision_macro": precision_score(
                    gold_idx, pred_idx, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    gold_idx, pred_idx, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    gold_idx, pred_idx, average="macro", zero_division=0
                ),
            }
        )

    metrics_df = pd.DataFrame.from_records(records)

    detail_df = pd.DataFrame(
        {
            "sentence1": df["sentence1"].tolist(),
            "sentence2": df["sentence2"].tolist(),
            "gold_label": df["gold_label"].tolist(),
            "gold_idx": gold_idx,
            "pred_idx_base": base_idx,
            "pred_idx_debiased": deb_idx,
            "pred_label_base": [idx2label[i] for i in base_idx],
            "pred_label_debiased": [idx2label[i] for i in deb_idx],
            "correct_base": [int(p == g) for p, g in zip(base_idx, gold_idx)],
            "correct_debiased": [int(p == g) for p, g in zip(deb_idx, gold_idx)],
            "lexical_overlap": lexical_overlaps,
        }
    )

    return metrics_df, detail_df


if __name__ == "__main__":
    model_prediction = ModelPrediction(
        MODEL_BASE_ID,
        MODEL_DEBIASED_ID,
        max_length=128,
    )

    metrics_frames: list[pd.DataFrame] = []
    analysis_frames: list[pd.DataFrame] = []
    for split_name, dataset_path in SYNONYM_DATASETS.items():
        df_synonym = _load_synonym_dataset(dataset_path)
        df_metrics, df_detail = _evaluate_predictions(df_synonym, model_prediction)
        df_metrics.insert(0, "split", split_name)
        metrics_frames.append(df_metrics)
        df_detail.insert(0, "split", split_name)
        analysis_frames.append(df_detail)

    metrics_df = pd.concat(metrics_frames, ignore_index=True)[
        [
            "split",
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ]

    with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
        print("Synonym Augmentation Evaluation Metrics")
        print(metrics_df.to_string(index=False))

    if analysis_frames:
        analysis_df = pd.concat(analysis_frames, ignore_index=True)

        analysis_df["overlap_bin"] = pd.cut(
            analysis_df["lexical_overlap"],
            bins=OVERLAP_BINS,
            labels=OVERLAP_BIN_LABELS,
            include_lowest=True,
            right=True,
        )

        overlap_summary = (
            analysis_df.groupby(["split", "overlap_bin"], dropna=False)
            .agg(
                count=("gold_idx", "size"),
                accuracy_base=("correct_base", "mean"),
                accuracy_debiased=("correct_debiased", "mean"),
            )
            .reset_index()
        )

        print("\nLexical Overlap vs Accuracy")
        for split_name, df_split in overlap_summary.groupby("split"):
            df_split = df_split[df_split["overlap_bin"].notna()].copy()
            if df_split.empty:
                continue
            df_split["overlap_bin"] = df_split["overlap_bin"].astype(str)
            with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
                print(f"\n{split_name}")
                print(
                    df_split[
                        [
                            "overlap_bin",
                            "count",
                            "accuracy_base",
                            "accuracy_debiased",
                        ]
                    ].to_string(index=False)
                )

        label_order = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
        idx_order = [label2idx[label] for label in label_order]

        print("\nConfusion Matrices")
        for split_name, df_split in analysis_df.groupby("split"):
            print(f"\n{split_name} - Base Model")
            cm_base = confusion_matrix(
                df_split["gold_idx"],
                df_split["pred_idx_base"],
                labels=idx_order,
            )
            cm_base_df = pd.DataFrame(cm_base, index=label_order, columns=label_order)
            print(cm_base_df.to_string())

            print(f"\n{split_name} - Debiased Model")
            cm_debiased = confusion_matrix(
                df_split["gold_idx"],
                df_split["pred_idx_debiased"],
                labels=idx_order,
            )
            cm_debiased_df = pd.DataFrame(
                cm_debiased, index=label_order, columns=label_order
            )
            print(cm_debiased_df.to_string())
