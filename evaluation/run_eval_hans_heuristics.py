from __future__ import annotations

from math import erf, sqrt
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from model_prediction import ModelPrediction


label2idx = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
pred2idx = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}


model_prediction = ModelPrediction(
    "ndsanjana/BERT-Base-MNLI-Orig",
    "ndsanjana/BERT-Base-MNLI-Debiased",
    max_length=128,
)


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _validate_predictions(
    preds: list[str],
    split_name: str,
    df: pd.DataFrame | None = None,
    model_tag: str | None = None,
) -> None:
    unknown = []
    for p in preds:
        if p in label2idx or p in pred2idx:
            continue
        if isinstance(p, str) and p.startswith("LABEL_"):
            try:
                int(p.split("_")[-1])
                continue
            except Exception:
                pass
        unknown.append(p)

    if not unknown:
        return

    os.makedirs("evaluation", exist_ok=True)
    tag = f"_{model_tag}" if model_tag else ""
    out_path = os.path.join(
        "evaluation", f"unknown_predictions_{_slugify(split_name)}{tag}.txt"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            "Unknown predicted labels encountered. Showing unique set (up to 200):\n"
        )
        uniq = sorted(set(map(str, unknown)))[:200]
        for u in uniq:
            fh.write(f"{u}\n")

    if df is not None:
        try:
            mask = [p in unknown for p in preds]
            sample = df.loc[mask, ["sentence1", "sentence2", "gold_label"]].copy()
            sample["pred"] = [p for p in preds if p in unknown][: len(sample)]
            sample_path = os.path.join(
                "evaluation",
                f"unknown_prediction_samples_{_slugify(split_name)}{tag}.csv",
            )
            sample.head(100).to_csv(sample_path, index=False)
        except Exception:
            pass

    raise ValueError(
        f"Unknown predicted labels in split '{split_name}'. Details written to {out_path}"
    )


def _to_idx_list(preds: list[str]) -> list[int]:
    out: list[int] = []
    unknown: list[str] = []
    for p in preds:
        if p in label2idx:
            out.append(label2idx[p])
        elif p in pred2idx:
            out.append(pred2idx[p])
        elif isinstance(p, str) and p.startswith("LABEL_"):
            try:
                out.append(int(p.split("_")[-1]))
            except Exception:
                unknown.append(p)
        else:
            unknown.append(p)
    if unknown:
        raise ValueError(
            "Unknown predicted labels encountered: "
            f"{sorted(set(map(str, unknown)))[:10]} (showing up to 10)"
        )
    return out


def _compute_pair_counts(
    df: pd.DataFrame, y_pred_base: list[str], y_pred_debiased: list[str]
) -> tuple[int, int, list[bool], list[bool]]:
    y_true_upper = df["gold_label"].astype(str).str.upper().tolist()
    unique_true = set(y_true_upper)

    pred_idx_base = _to_idx_list(y_pred_base)
    pred_idx_debiased = _to_idx_list(y_pred_debiased)

    base_correct: list[bool] = []
    debiased_correct: list[bool] = []

    if unique_true <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        for yt, pb, pd in zip(y_true_upper, pred_idx_base, pred_idx_debiased):
            b_ok = (yt == "ENTAILMENT" and pb == label2idx["ENTAILMENT"]) or (
                yt == "NON-ENTAILMENT" and pb != label2idx["ENTAILMENT"]
            )
            d_ok = (yt == "ENTAILMENT" and pd == label2idx["ENTAILMENT"]) or (
                yt == "NON-ENTAILMENT" and pd != label2idx["ENTAILMENT"]
            )
            base_correct.append(b_ok)
            debiased_correct.append(d_ok)
    else:
        y_true_idx = [label2idx[yt] for yt in y_true_upper]
        for yt, pb, pd in zip(y_true_idx, pred_idx_base, pred_idx_debiased):
            base_correct.append(pb == yt)
            debiased_correct.append(pd == yt)

    b = int(
        sum(
            1 for b_ok, d_ok in zip(base_correct, debiased_correct) if b_ok and not d_ok
        )
    )
    c = int(
        sum(
            1
            for b_ok, d_ok in zip(base_correct, debiased_correct)
            if (not b_ok) and d_ok
        )
    )

    return b, c, base_correct, debiased_correct


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    z = (abs(b - c) - 1.0) / sqrt(n) if n > 0 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(z))
    return max(min(p, 1.0), 0.0)


def _bootstrap_ci(
    y_true_idx: list[int],
    pred_idx_base: list[int],
    pred_idx_debiased: list[int],
    num_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(random_seed)
    n = len(y_true_idx)
    acc_deltas: list[float] = []
    f1_deltas: list[float] = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = [y_true_idx[i] for i in idx]
        pb = [pred_idx_base[i] for i in idx]
        pd = [pred_idx_debiased[i] for i in idx]
        acc_b = accuracy_score(yb, pb)
        acc_d = accuracy_score(yb, pd)
        f1_b = f1_score(yb, pb, average="macro")
        f1_d = f1_score(yb, pd, average="macro")
        acc_deltas.append(acc_b - acc_d)
        f1_deltas.append(f1_b - f1_d)

    def _ci(a: list[float]) -> tuple[float, float]:
        low, high = np.percentile(a, [2.5, 97.5]).tolist()
        return float(low), float(high)

    acc_low, acc_high = _ci(acc_deltas)
    f1_low, f1_high = _ci(f1_deltas)
    return {
        "acc_delta_ci": (acc_low, acc_high),
        "f1_delta_ci": (f1_low, f1_high),
        "acc_delta_mean": float(np.mean(acc_deltas)),
        "f1_delta_mean": float(np.mean(f1_deltas)),
    }


def _prep_bootstrap_inputs(
    df: pd.DataFrame, y_pred_base: list[str], y_pred_debiased: list[str]
) -> tuple[list[int], list[int], list[int]]:
    y_true_upper = df["gold_label"].astype(str).str.upper().tolist()
    if set(y_true_upper) <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        y_true_idx = [0 if y == "ENTAILMENT" else 1 for y in y_true_upper]

        def _to_bin(preds: list[str]) -> list[int]:
            out: list[int] = []
            for p in preds:
                if p in label2idx:
                    out.append(0 if p == "ENTAILMENT" else 1)
                elif p in pred2idx:
                    out.append(0 if pred2idx[p] == label2idx["ENTAILMENT"] else 1)
                elif isinstance(p, str) and p.startswith("LABEL_"):
                    idx = int(p.split("_")[-1])
                    out.append(0 if idx == label2idx["ENTAILMENT"] else 1)
                else:
                    raise ValueError(f"Unknown predicted label encountered: {p}")
            return out

        pred_idx_base = _to_bin(y_pred_base)
        pred_idx_debiased = _to_bin(y_pred_debiased)
    else:
        y_true_idx = [label2idx[y] for y in y_true_upper]
        pred_idx_base = _to_idx_list(y_pred_base)
        pred_idx_debiased = _to_idx_list(y_pred_debiased)

    return y_true_idx, pred_idx_base, pred_idx_debiased


def get_predictions(df: pd.DataFrame, desc: str | None = None):
    y_pred_base: list[str] = []
    y_pred_debiased: list[str] = []
    prob_base: list[list[float]] = []
    prob_debiased: list[list[float]] = []
    logits_base: list[list[float]] = []
    logits_debiased: list[list[float]] = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=desc or "Predicting", unit="ex"
    ):
        premise = row["sentence1"]
        hypothesis = row["sentence2"]
        input_base, input_debiased = model_prediction.tokenize_input(
            premise, hypothesis
        )
        (
            pred_base,
            pred_debiased,
            p_base,
            p_debiased,
            l_base,
            l_debiased,
        ) = model_prediction.predict(input_base, input_debiased)
        y_pred_base.append(pred_base)
        y_pred_debiased.append(pred_debiased)
        prob_base.append(p_base)
        prob_debiased.append(p_debiased)
        logits_base.append(l_base)
        logits_debiased.append(l_debiased)

    return (
        y_pred_base,
        y_pred_debiased,
        prob_base,
        prob_debiased,
        logits_base,
        logits_debiased,
    )


def compute_metrics(
    y_pred_base: list[str], y_pred_debiased: list[str], y_true: list[str]
) -> dict:
    y_true_upper = [y.upper() for y in y_true]
    unique_labels = set(y_true_upper)

    y_pred_base_idx = _to_idx_list(y_pred_base)
    y_pred_debiased_idx = _to_idx_list(y_pred_debiased)

    if unique_labels <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        y_true_bin = [0 if y == "ENTAILMENT" else 1 for y in y_true_upper]
        y_pred_base_bin = [
            0 if i == label2idx["ENTAILMENT"] else 1 for i in y_pred_base_idx
        ]
        y_pred_debiased_bin = [
            0 if i == label2idx["ENTAILMENT"] else 1 for i in y_pred_debiased_idx
        ]

        accuracy_base = accuracy_score(y_true_bin, y_pred_base_bin)
        accuracy_debiased = accuracy_score(y_true_bin, y_pred_debiased_bin)
        f1_base = f1_score(y_true_bin, y_pred_base_bin, average="macro")
        f1_debiased = f1_score(y_true_bin, y_pred_debiased_bin, average="macro")
        precision_base = precision_score(
            y_true_bin, y_pred_base_bin, average="macro", zero_division=0
        )
        precision_debiased = precision_score(
            y_true_bin, y_pred_debiased_bin, average="macro", zero_division=0
        )
        recall_base = recall_score(
            y_true_bin, y_pred_base_bin, average="macro", zero_division=0
        )
        recall_debiased = recall_score(
            y_true_bin, y_pred_debiased_bin, average="macro", zero_division=0
        )
        labels_order_idx = [0, 1]
        cm_base = confusion_matrix(y_true_bin, y_pred_base_bin, labels=labels_order_idx)
        cm_debiased = confusion_matrix(
            y_true_bin, y_pred_debiased_bin, labels=labels_order_idx
        )
    else:
        allowed = set(label2idx.keys()) | {"NON-ENTAILMENT"}
        bad = [y for y in y_true_upper if y not in allowed]
        if bad:
            raise ValueError(
                f"Unknown gold labels: {sorted(set(bad))[:10]} (showing up to 10)"
            )

        y_true_idx = [label2idx[y] for y in y_true_upper]
        accuracy_base = accuracy_score(y_true_idx, y_pred_base_idx)
        accuracy_debiased = accuracy_score(y_true_idx, y_pred_debiased_idx)
        f1_base = f1_score(y_true_idx, y_pred_base_idx, average="macro")
        f1_debiased = f1_score(y_true_idx, y_pred_debiased_idx, average="macro")
        precision_base = precision_score(
            y_true_idx, y_pred_base_idx, average="macro", zero_division=0
        )
        precision_debiased = precision_score(
            y_true_idx, y_pred_debiased_idx, average="macro", zero_division=0
        )
        recall_base = recall_score(
            y_true_idx, y_pred_base_idx, average="macro", zero_division=0
        )
        recall_debiased = recall_score(
            y_true_idx, y_pred_debiased_idx, average="macro", zero_division=0
        )
        labels_order_idx = [0, 1, 2]
        cm_base = confusion_matrix(y_true_idx, y_pred_base_idx, labels=labels_order_idx)
        cm_debiased = confusion_matrix(
            y_true_idx, y_pred_debiased_idx, labels=labels_order_idx
        )

    return {
        "base": {
            "accuracy": accuracy_base,
            "f1_macro": f1_base,
            "precision_macro": precision_base,
            "recall_macro": recall_base,
            "confusion": cm_base,
        },
        "debiased": {
            "accuracy": accuracy_debiased,
            "f1_macro": f1_debiased,
            "precision_macro": precision_debiased,
            "recall_macro": recall_debiased,
            "confusion": cm_debiased,
        },
    }


def _format_confusion_matrix(cm: np.ndarray, labels_order: list[int]) -> str:
    df_cm = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    df_cm.index.name = "true"
    return df_cm.to_string()


if __name__ == "__main__":
    os.makedirs("evaluation", exist_ok=True)

    hans = pd.read_json("data/HANS/heuristics_evaluation_set.jsonl", lines=True)
    hans_labels_set = set(hans["gold_label"].astype(str).str.upper().unique())
    assert hans_labels_set <= {
        "ENTAILMENT",
        "NON-ENTAILMENT",
    }, f"HANS gold labels invalid: {hans_labels_set}"

    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(hans, desc="HANS (heuristics)")

    _validate_predictions(y_pred_base, "HANS", df=hans, model_tag="base")
    _validate_predictions(y_pred_debiased, "HANS", df=hans, model_tag="debiased")

    hans = hans.assign(
        pred_base=y_pred_base,
        pred_debiased=y_pred_debiased,
        prob_base=p_base,
        prob_debiased=p_debiased,
        logit_base=l_base,
        logit_debiased=l_debiased,
    )

    results_rows: list[dict] = []
    confusion_outputs: list[dict] = []
    label_summaries: list[dict] = []

    stat_lines = ["Statistical tests per heuristic (base vs debiased)"]

    heuristics = sorted(hans["heuristic"].astype(str).unique())
    labels_order_binary = [0, 1]

    for heur in heuristics:
        df_h = hans[hans["heuristic"] == heur].copy()
        y_true = df_h["gold_label"].tolist()
        y_base = df_h["pred_base"].tolist()
        y_deb = df_h["pred_debiased"].tolist()

        metrics = compute_metrics(y_base, y_deb, y_true)

        b, c, base_ok, deb_ok = _compute_pair_counts(df_h, y_base, y_deb)
        y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
            df_h, y_base, y_deb
        )
        p_value = _mcnemar_p_value(b, c)
        boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
        stat_lines.append(
            (
                f"[{heur}] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
                f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
                f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}"
            )
        )

        results_rows.append(
            {
                "heuristic": heur,
                "model": "base",
                "accuracy": metrics["base"]["accuracy"],
                "f1_macro": metrics["base"]["f1_macro"],
                "precision_macro": metrics["base"]["precision_macro"],
                "recall_macro": metrics["base"]["recall_macro"],
                "n": len(df_h),
                "b_base_only_correct": b,
                "c_debiased_only_correct": c,
            }
        )
        results_rows.append(
            {
                "heuristic": heur,
                "model": "debiased",
                "accuracy": metrics["debiased"]["accuracy"],
                "f1_macro": metrics["debiased"]["f1_macro"],
                "precision_macro": metrics["debiased"]["precision_macro"],
                "recall_macro": metrics["debiased"]["recall_macro"],
                "n": len(df_h),
                "b_base_only_correct": b,
                "c_debiased_only_correct": c,
            }
        )

        confusion_outputs.append(
            {
                "heuristic": heur,
                "model": "base",
                "labels": labels_order_binary,
                "cm": metrics["base"]["confusion"],
            }
        )
        confusion_outputs.append(
            {
                "heuristic": heur,
                "model": "debiased",
                "labels": labels_order_binary,
                "cm": metrics["debiased"]["confusion"],
            }
        )

        label_summaries.append(
            {
                "heuristic": heur,
                "n": len(df_h),
                "counts": df_h["gold_label"]
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
                .to_dict(),
            }
        )

    results_df = pd.DataFrame(results_rows)[
        [
            "heuristic",
            "model",
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "n",
            "b_base_only_correct",
            "c_debiased_only_correct",
        ]
    ]

    with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
        table_str = results_df.to_string(index=False)
        print(table_str)

    results_path = "evaluation/hans_heuristic_results.txt"
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write(table_str)
        fh.write("\n")

    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write("\n\nConfusion matrices (rows=true, cols=pred):\n")
        for item in confusion_outputs:
            fh.write(f"\n[{item['heuristic']}] - {item['model']}\n")
            fh.write(_format_confusion_matrix(item["cm"], item["labels"]))
            fh.write("\n")

        fh.write("\nLabel distributions (lowercased):\n")
        for summary in label_summaries:
            fh.write(f"\n[{summary['heuristic']}] n={summary['n']}\n")
            order = ["entailment", "neutral", "contradiction"]
            counts = summary["counts"]
            for key in order:
                if key in counts:
                    fh.write(f"  {key}: {counts[key]}\n")
            for key, value in counts.items():
                if key not in order:
                    fh.write(f"  {key}: {value}\n")

    stat_path = "evaluation/hans_heuristic_stat_tests.txt"
    with open(stat_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(stat_lines))
        fh.write("\n")

    print(f"Saved results to {results_path}")
    print(f"Saved statistical tests to {stat_path}")
