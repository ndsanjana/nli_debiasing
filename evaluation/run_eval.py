from model_prediction import ModelPrediction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import json
import numpy as np
from math import sqrt, erf


model_prediction = ModelPrediction(
    "ndsanjana/BERT-Base-MNLI-Orig", "ndsanjana/BERT-Base-MNLI-Debiased", max_length=128
)
label2idx = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
pred2idx = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}


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
    if unknown:
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
        # Optional: dump sample problematic examples
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


def _save_predictions(
    split_name: str,
    data_frame: pd.DataFrame,
    y_pred_base: list[str],
    y_pred_debiased: list[str],
    prob_base: list[list[float]],
    prob_debiased: list[list[float]],
    logits_base: list[list[float]] | None = None,
    logits_debiased: list[list[float]] | None = None,
) -> None:
    """Save per-example predictions and probabilities for analysis."""
    os.makedirs("evaluation/preds", exist_ok=True)
    os.makedirs("evaluation/pairs", exist_ok=True)
    label_names = [k for k, v in sorted(label2idx.items(), key=lambda kv: kv[1])]

    df_out = data_frame[["sentence1", "sentence2", "gold_label"]].copy()
    # Stable example id from source index
    df_out["example_id"] = data_frame.index
    df_out["pred_base"] = y_pred_base
    df_out["pred_debiased"] = y_pred_debiased

    # Validate probability/logit vector lengths match canonical order
    if not all(len(p) == len(label_names) for p in prob_base + prob_debiased):
        raise ValueError(
            f"Probability vector length mismatch. Expected {len(label_names)} for labels {label_names}."
        )
    if logits_base is not None and logits_debiased is not None:
        if not all(len(v) == len(label_names) for v in logits_base + logits_debiased):
            raise ValueError(
                f"Logit vector length mismatch. Expected {len(label_names)} for labels {label_names}."
            )

    # Expand probability vectors into columns using the canonical NLI order
    for i, lbl in enumerate(label_names):
        df_out[f"prob_base_{lbl}"] = [
            probs[i] if i < len(probs) else None for probs in prob_base
        ]
        df_out[f"prob_debiased_{lbl}"] = [
            probs[i] if i < len(probs) else None for probs in prob_debiased
        ]

    # Optionally save logits too
    if logits_base is not None and logits_debiased is not None:
        for i, lbl in enumerate(label_names):
            df_out[f"logit_base_{lbl}"] = [
                values[i] if i < len(values) else None for values in logits_base
            ]
            df_out[f"logit_debiased_{lbl}"] = [
                values[i] if i < len(values) else None for values in logits_debiased
            ]

    # Compute correctness flags and paired table for McNemar
    y_true_upper = data_frame["gold_label"].astype(str).str.upper().tolist()
    unique_true = set(y_true_upper)

    # Map predictions to idx (3-class indices)
    def _to_idx_list(preds: list[str]) -> list[int]:
        out = []
        for p in preds:
            if p in label2idx:
                out.append(label2idx[p])
            elif p in pred2idx:
                out.append(pred2idx[p])
            elif isinstance(p, str) and p.startswith("LABEL_"):
                out.append(int(p.split("_")[-1]))
            else:
                raise ValueError(f"Unknown predicted label encountered: {p}")
        return out

    pred_idx_base = _to_idx_list(y_pred_base)
    pred_idx_debiased = _to_idx_list(y_pred_debiased)

    # Compute correctness according to dataset type
    base_correct, debiased_correct = [], []
    if unique_true <= {"ENTAILMENT", "NON-ENTAILMENT"}:  # HANS
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

    df_out["base_correct"] = base_correct
    df_out["debiased_correct"] = debiased_correct

    # Save paired table and counts
    out_file = os.path.join("evaluation/preds", f"{_slugify(split_name)}.csv")
    df_out.to_csv(out_file, index=False)

    paired_csv = os.path.join(
        "evaluation/pairs", f"{_slugify(split_name)}_paired_table.csv"
    )
    df_pairs = df_out[
        [
            "example_id",
            "gold_label",
            "pred_base",
            "pred_debiased",
            "base_correct",
            "debiased_correct",
        ]
    ].copy()
    df_pairs.to_csv(paired_csv, index=False)

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
    counts_path = os.path.join(
        "evaluation/pairs", f"{_slugify(split_name)}_paired_counts.json"
    )
    with open(counts_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"b_base_only_correct": b, "c_debiased_only_correct": c}, fh, indent=2
        )

    return b, c, base_correct, debiased_correct


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _mcnemar_p_value(b: int, c: int) -> float:
    # Normal approximation with continuity correction
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
    acc_deltas, f1_deltas = [], []
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

    def _ci(a: list[float]):
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


def _validate_id2label_order(mp: ModelPrediction) -> None:
    os.makedirs("evaluation", exist_ok=True)
    canonical_order = [k for k, v in sorted(label2idx.items(), key=lambda kv: kv[1])]
    base_map = {int(k): str(v).upper() for k, v in mp.config_base.id2label.items()}
    deb_map = {int(k): str(v).upper() for k, v in mp.config_debiased.id2label.items()}
    base_order = [base_map.get(i, f"IDX_{i}") for i in range(len(canonical_order))]
    deb_order = [deb_map.get(i, f"IDX_{i}") for i in range(len(canonical_order))]
    out = {
        "canonical_label2idx": label2idx,
        "canonical_idx2label": {v: k for k, v in label2idx.items()},
        "base_id2label": base_map,
        "debiased_id2label": deb_map,
    }
    with open("evaluation/label_index_map.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    def _is_generic(order: list[str]) -> bool:
        return all(isinstance(s, str) and s.upper().startswith("LABEL_") for s in order)

    if base_order == canonical_order and deb_order == canonical_order:
        return
    if _is_generic(base_order) or _is_generic(deb_order):
        print(
            "Warning: id2label contains generic LABEL_i entries; assuming indices 0/1/2 map to canonical ENTAILMENT/NEUTRAL/CONTRADICTION. See evaluation/label_index_map.json"
        )
        return
    # Textual labels present but order differs. Mapping will be handled via textual labels in predictions.
    print(
        f"Warning: Model id2label ordering differs from canonical. Canonical {canonical_order}, base {base_order}, debiased {deb_order}. Using textual labels for mapping."
    )
    return


def _prep_bootstrap_inputs(
    df: pd.DataFrame, y_pred_base: list[str], y_pred_debiased: list[str]
) -> tuple[list[int], list[int], list[int]]:
    y_true_upper = df["gold_label"].astype(str).str.upper().tolist()
    if set(y_true_upper) <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        y_true_idx = [0 if y == "ENTAILMENT" else 1 for y in y_true_upper]

        def _to_bin(preds: list[str]) -> list[int]:
            out = []
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

        def _to_idx(preds: list[str]) -> list[int]:
            out = []
            for p in preds:
                if p in label2idx:
                    out.append(label2idx[p])
                elif p in pred2idx:
                    out.append(pred2idx[p])
                elif isinstance(p, str) and p.startswith("LABEL_"):
                    out.append(int(p.split("_")[-1]))
                else:
                    raise ValueError(f"Unknown predicted label encountered: {p}")
            return out

        pred_idx_base = _to_idx(y_pred_base)
        pred_idx_debiased = _to_idx(y_pred_debiased)
    return y_true_idx, pred_idx_base, pred_idx_debiased


def get_predictions(df: pd.DataFrame, desc: str | None = None):
    y_pred_base, y_pred_debiased = [], []
    prob_base, prob_debiased = [], []
    logits_base, logits_debiased = [], []
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


def compute_metrics(y_pred_base, y_pred_debiased, y_true):
    y_true_upper = [y.upper() for y in y_true]
    unique_labels = set(y_true_upper)

    # Map predicted string labels to 3-class indices (robust to either LABEL_X or textual labels)
    def _to_idx_list(preds):
        out = []
        unknown_labels = []
        for p in preds:
            if p in label2idx:
                out.append(label2idx[p])
            elif p in pred2idx:
                out.append(pred2idx[p])
            elif isinstance(p, str) and p.startswith("LABEL_"):
                try:
                    out.append(int(p.split("_")[-1]))
                except Exception:
                    unknown_labels.append(p)
            else:
                unknown_labels.append(p)
        if unknown_labels:
            raise ValueError(
                f"Unknown predicted labels encountered: {sorted(set(map(str, unknown_labels)))[:10]} (showing up to 10)"
            )
        return out

    y_pred_base_idx = _to_idx_list(y_pred_base)
    y_pred_debiased_idx = _to_idx_list(y_pred_debiased)

    if unique_labels <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        # Binary collapse for HANS: entailment (0) vs non-entailment (1)
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

        # 2x2 confusion matrices (rows=true, cols=pred) with labels [0,1]
        labels_order_idx = [0, 1]
        cm_base = confusion_matrix(y_true_bin, y_pred_base_bin, labels=labels_order_idx)
        cm_debiased = confusion_matrix(
            y_true_bin, y_pred_debiased_bin, labels=labels_order_idx
        )
    else:
        # Standard 3-class NLI
        # Validate gold labels are known
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

        # 3x3 confusion matrices (rows=true, cols=pred), fixed label order 0,1,2
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


def _format_confusion_matrix(cm, labels_order):
    df_cm = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    df_cm.index.name = "true"
    return df_cm.to_string()


def _read_augmented_trsf(path: str) -> pd.DataFrame:
    """Read augmented transformed TSVs with validation.

    Supports either:
    - Explicit header containing columns `sentence1`, `sentence2`, `gold_label`
    - Or infers these as last 4th, last 3rd, and last columns respectively, with assertions
    """
    try:
        df_try = pd.read_csv(
            path, sep="\t", header=0, dtype=str, encoding="utf-8", on_bad_lines="skip"
        )
    except Exception:
        df_try = None

    if df_try is not None and {"sentence1", "sentence2", "gold_label"}.issubset(
        df_try.columns
    ):
        out = df_try[["sentence1", "sentence2", "gold_label"]].copy()
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            dtype=str,
            encoding="utf-8",
            on_bad_lines="skip",
        )
        if df.shape[1] < 4:
            raise ValueError(
                f"Unexpected format for {path}: requires at least 4 columns, found {df.shape[1]}"
            )
        premise_col = df.columns[-4]
        hypothesis_col = df.columns[-3]
        label_col = df.columns[-1]
        out = df[[premise_col, hypothesis_col, label_col]].copy()
        out.columns = ["sentence1", "sentence2", "gold_label"]

    # Normalize and validate labels
    out["sentence1"] = out["sentence1"].astype(str).str.strip()
    out["sentence2"] = out["sentence2"].astype(str).str.strip()
    out["gold_label"] = out["gold_label"].astype(str).str.strip().str.upper()
    out = out[(out["sentence1"] != "") & (out["sentence2"] != "")]

    allowed = {"ENTAILMENT", "NEUTRAL", "CONTRADICTION", "NON-ENTAILMENT"}
    unexpected = sorted(set(out["gold_label"]) - allowed)
    if unexpected:
        # Dump a small sample for inspection
        sample_path = path + ".bad_labels.sample.csv"
        out[out["gold_label"].isin(unexpected)].head(50).to_csv(
            sample_path, index=False
        )
        raise ValueError(
            f"Unexpected gold_label values in {path}: {unexpected}. Sample saved to {sample_path}"
        )
    return out


def _read_augmented_orig(path: str) -> pd.DataFrame:
    # Original augmented files share the same structure as transformed
    return _read_augmented_trsf(path)


if __name__ == "__main__":
    rows = []
    confusion_outputs = []
    label_summaries = []
    # Validate model id2label ordering and write label index mapping
    _validate_id2label_order(model_prediction)
    # Initialize statistical tests output file
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/stat_tests.txt", "w", encoding="utf-8") as fh:
        fh.write("Statistical tests per split (base vs debiased)\n")

    # HANS (binary)
    hans_data = pd.read_json("data/HANS/heuristics_evaluation_set.jsonl", lines=True)
    # Validate HANS labels explicitly
    hans_labels_set = set(hans_data["gold_label"].astype(str).str.upper().unique())
    assert hans_labels_set <= {
        "ENTAILMENT",
        "NON-ENTAILMENT",
    }, f"HANS gold labels invalid: {hans_labels_set}"
    y_pred_base, y_pred_debiased, p_base, p_debiased, l_base, l_debiased = (
        get_predictions(hans_data, desc="HANS")
    )
    _validate_predictions(y_pred_base, "HANS", df=hans_data, model_tag="base")
    _validate_predictions(y_pred_debiased, "HANS", df=hans_data, model_tag="debiased")
    # Print uniques for visibility
    print("HANS unique gold labels:", sorted(hans_labels_set))
    print("HANS unique predicted labels (base):", sorted(set(y_pred_base)))
    print("HANS unique predicted labels (debiased):", sorted(set(y_pred_debiased)))
    m = compute_metrics(y_pred_base, y_pred_debiased, hans_data["gold_label"])
    # Statistical tests
    # Prepare inputs for bootstrap
    y_true_upper = hans_data["gold_label"].astype(str).str.upper().tolist()
    if set(y_true_upper) <= {"ENTAILMENT", "NON-ENTAILMENT"}:
        # Collapse to binary indices for accuracy only
        y_true_idx = [0 if y == "ENTAILMENT" else 1 for y in y_true_upper]

        # Convert preds to binary by collapse
        def _to_bin(preds):
            # Map preds to index then collapse
            tmp = []
            for p in preds:
                if p in label2idx:
                    tmp.append(0 if p == "ENTAILMENT" else 1)
                elif p in pred2idx:
                    tmp.append(0 if pred2idx[p] == label2idx["ENTAILMENT"] else 1)
                elif isinstance(p, str) and p.startswith("LABEL_"):
                    idx = int(p.split("_")[-1])
                    tmp.append(0 if idx == label2idx["ENTAILMENT"] else 1)
                else:
                    raise ValueError(f"Unknown predicted label encountered: {p}")
            return tmp

        pred_idx_base = _to_bin(y_pred_base)
        pred_idx_debiased = _to_bin(y_pred_debiased)
    else:
        # Standard 3-class
        y_true_idx = [label2idx[y] for y in y_true_upper]

        def _to_idx(preds):
            out = []
            for p in preds:
                if p in label2idx:
                    out.append(label2idx[p])
                elif p in pred2idx:
                    out.append(pred2idx[p])
                elif isinstance(p, str) and p.startswith("LABEL_"):
                    out.append(int(p.split("_")[-1]))
                else:
                    raise ValueError(f"Unknown predicted label encountered: {p}")
            return out

        pred_idx_base = _to_idx(y_pred_base)
        pred_idx_debiased = _to_idx(y_pred_debiased)

    b, c, base_ok, deb_ok = _save_predictions(
        "HANS",
        hans_data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"\n[HANS] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append({"split": "HANS", "model": "base", **m["base"], "n": len(hans_data)})
    rows.append(
        {"split": "HANS", "model": "debiased", **m["debiased"], "n": len(hans_data)}
    )
    hans_labels = ["ENTAILMENT", "NON-ENTAILMENT"]
    confusion_outputs.append(
        {
            "split": "HANS",
            "model": "base",
            "labels": hans_labels,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "HANS",
            "model": "debiased",
            "labels": hans_labels,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Negated Dev Matched
    data = pd.read_csv("data/mnli/negated/mnli_dev_matched_1nt.tsv", sep="\t")
    label_summaries.append(
        {
            "split": "Negated Dev Matched",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Negated Dev Matched")
    # Print uniques and validate
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    assert gold_set <= {
        "ENTAILMENT",
        "NEUTRAL",
        "CONTRADICTION",
    }, f"Negated Dev Matched gold labels invalid: {gold_set}"
    _validate_predictions(y_pred_base, "Negated Dev Matched", df=data, model_tag="base")
    _validate_predictions(
        y_pred_debiased, "Negated Dev Matched", df=data, model_tag="debiased"
    )
    print("Negated Dev Matched unique gold labels:", sorted(gold_set))
    print(
        "Negated Dev Matched unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Negated Dev Matched unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Negated Dev Matched",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Negated Dev Matched] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {"split": "Negated Dev Matched", "model": "base", **m["base"], "n": len(data)}
    )
    rows.append(
        {
            "split": "Negated Dev Matched",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    labels_order = [k for k, v in sorted(label2idx.items(), key=lambda kv: kv[1])]
    confusion_outputs.append(
        {
            "split": "Negated Dev Matched",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Negated Dev Matched",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Negated Dev Mismatched
    data = pd.read_csv("data/mnli/negated/mnli_dev_mismatched_1nt.tsv", sep="\t")
    label_summaries.append(
        {
            "split": "Negated Dev Mismatched",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Negated Dev Mismatched")
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    assert gold_set <= {
        "ENTAILMENT",
        "NEUTRAL",
        "CONTRADICTION",
    }, f"Negated Dev Mismatched gold labels invalid: {gold_set}"
    _validate_predictions(
        y_pred_base, "Negated Dev Mismatched", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Negated Dev Mismatched", df=data, model_tag="debiased"
    )
    print("Negated Dev Mismatched unique gold labels:", sorted(gold_set))
    print(
        "Negated Dev Mismatched unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Negated Dev Mismatched unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Negated Dev Mismatched",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Negated Dev Mismatched] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Negated Dev Mismatched",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Negated Dev Mismatched",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Negated Dev Mismatched",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Negated Dev Mismatched",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Combined Augmented Transformed
    data = _read_augmented_trsf("data/Augmented/trsf/comb_trsf_large.tsv")
    label_summaries.append(
        {
            "split": "Combined Augmented Transformed",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Combined Augmented Transformed")
    _validate_predictions(
        y_pred_base, "Combined Augmented Transformed", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Combined Augmented Transformed", df=data, model_tag="debiased"
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Combined Augmented Transformed unique gold labels:", sorted(gold_set))
    print(
        "Combined Augmented Transformed unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Combined Augmented Transformed unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Combined Augmented Transformed",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Combined Augmented Transformed] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Combined Augmented Transformed",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Combined Augmented Transformed",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Combined Augmented Transformed",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Combined Augmented Transformed",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Passivized Augmented Transformed
    data = _read_augmented_trsf("data/Augmented/trsf/pass_trsf_large.tsv")
    label_summaries.append(
        {
            "split": "Passivized Augmented Transformed",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Passivized Augmented Transformed")
    _validate_predictions(
        y_pred_base, "Passivized Augmented Transformed", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased,
        "Passivized Augmented Transformed",
        df=data,
        model_tag="debiased",
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Passivized Augmented Transformed unique gold labels:", sorted(gold_set))
    print(
        "Passivized Augmented Transformed unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Passivized Augmented Transformed unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Passivized Augmented Transformed",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Passivized Augmented Transformed] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Passivized Augmented Transformed",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Passivized Augmented Transformed",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Passivized Augmented Transformed",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Passivized Augmented Transformed",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Inverted Augmented Transformed
    data = _read_augmented_trsf("data/Augmented/trsf/inv_trsf_large.tsv")
    label_summaries.append(
        {
            "split": "Inverted Augmented Transformed",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Inverted Augmented Transformed")
    _validate_predictions(
        y_pred_base, "Inverted Augmented Transformed", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Inverted Augmented Transformed", df=data, model_tag="debiased"
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Inverted Augmented Transformed unique gold labels:", sorted(gold_set))
    print(
        "Inverted Augmented Transformed unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Inverted Augmented Transformed unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Inverted Augmented Transformed",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Inverted Augmented Transformed] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Inverted Augmented Transformed",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Inverted Augmented Transformed",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )

    # Combined Augmented Original
    data = _read_augmented_orig("data/Augmented/orig/comb_orig_large.tsv")
    label_summaries.append(
        {
            "split": "Combined Augmented Original",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Combined Augmented Original")
    _validate_predictions(
        y_pred_base, "Combined Augmented Original", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Combined Augmented Original", df=data, model_tag="debiased"
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Combined Augmented Original unique gold labels:", sorted(gold_set))
    print(
        "Combined Augmented Original unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Combined Augmented Original unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Combined Augmented Original",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Combined Augmented Original] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Combined Augmented Original",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Combined Augmented Original",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Combined Augmented Original",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Combined Augmented Original",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Passivized Augmented Original
    data = _read_augmented_orig("data/Augmented/orig/pass_orig_large.tsv")
    label_summaries.append(
        {
            "split": "Passivized Augmented Original",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Passivized Augmented Original")
    _validate_predictions(
        y_pred_base, "Passivized Augmented Original", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Passivized Augmented Original", df=data, model_tag="debiased"
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Passivized Augmented Original unique gold labels:", sorted(gold_set))
    print(
        "Passivized Augmented Original unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Passivized Augmented Original unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Passivized Augmented Original",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Passivized Augmented Original] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Passivized Augmented Original",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Passivized Augmented Original",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Passivized Augmented Original",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Passivized Augmented Original",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Inverted Augmented Original
    data = _read_augmented_orig("data/Augmented/orig/inv_orig_large.tsv")
    label_summaries.append(
        {
            "split": "Inverted Augmented Original",
            "n": len(data),
            "counts": data["gold_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict(),
        }
    )
    (
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    ) = get_predictions(data, desc="Inverted Augmented Original")
    _validate_predictions(
        y_pred_base, "Inverted Augmented Original", df=data, model_tag="base"
    )
    _validate_predictions(
        y_pred_debiased, "Inverted Augmented Original", df=data, model_tag="debiased"
    )
    gold_set = set(data["gold_label"].astype(str).str.upper().unique())
    print("Inverted Augmented Original unique gold labels:", sorted(gold_set))
    print(
        "Inverted Augmented Original unique predicted labels (base):",
        sorted(set(y_pred_base)),
    )
    print(
        "Inverted Augmented Original unique predicted labels (debiased):",
        sorted(set(y_pred_debiased)),
    )
    m = compute_metrics(y_pred_base, y_pred_debiased, data["gold_label"])
    b, c, _, _ = _save_predictions(
        "Inverted Augmented Original",
        data,
        y_pred_base,
        y_pred_debiased,
        p_base,
        p_debiased,
        l_base,
        l_debiased,
    )
    y_true_idx, pred_idx_base, pred_idx_debiased = _prep_bootstrap_inputs(
        data, y_pred_base, y_pred_debiased
    )
    p_value = _mcnemar_p_value(b, c)
    boot = _bootstrap_ci(y_true_idx, pred_idx_base, pred_idx_debiased)
    with open("evaluation/stat_tests.txt", "a", encoding="utf-8") as fh:
        fh.write(
            f"[Inverted Augmented Original] McNemar b={b}, c={c}, p-value={p_value:.6g}; "
            f"Acc Δ mean={boot['acc_delta_mean']:.6f} CI95={boot['acc_delta_ci']}; "
            f"F1 Δ mean={boot['f1_delta_mean']:.6f} CI95={boot['f1_delta_ci']}\n"
        )
    rows.append(
        {
            "split": "Inverted Augmented Original",
            "model": "base",
            **m["base"],
            "n": len(data),
        }
    )
    rows.append(
        {
            "split": "Inverted Augmented Original",
            "model": "debiased",
            **m["debiased"],
            "n": len(data),
        }
    )
    confusion_outputs.append(
        {
            "split": "Inverted Augmented Original",
            "model": "base",
            "labels": labels_order,
            "cm": m["base"]["confusion"],
        }
    )
    confusion_outputs.append(
        {
            "split": "Inverted Augmented Original",
            "model": "debiased",
            "labels": labels_order,
            "cm": m["debiased"]["confusion"],
        }
    )

    # Weighted aggregates (exclude HANS)
    def _add_weighted_aggregate(name: str, include_splits: list[str]):
        for model in ("base", "debiased"):
            subset = [
                r for r in rows if r["model"] == model and r["split"] in include_splits
            ]
            n_total = sum(r.get("n", 0) for r in subset)
            if n_total == 0:
                continue

            def wavg(key: str) -> float:
                return sum(r[key] * r.get("n", 0) for r in subset) / n_total

            rows.append(
                {
                    "split": name,
                    "model": model,
                    "accuracy": wavg("accuracy"),
                    "f1_macro": wavg("f1_macro"),
                    "precision_macro": wavg("precision_macro"),
                    "recall_macro": wavg("recall_macro"),
                    "n": n_total,
                }
            )

    negated_splits = ["Negated Dev Matched", "Negated Dev Mismatched"]
    trsf_splits = [
        "Combined Augmented Transformed",
        "Passivized Augmented Transformed",
        "Inverted Augmented Transformed",
    ]
    orig_splits = [
        "Combined Augmented Original",
        "Passivized Augmented Original",
        "Inverted Augmented Original",
    ]
    _add_weighted_aggregate(
        "Weighted Negated + Transformed", negated_splits + trsf_splits
    )
    _add_weighted_aggregate("Weighted Negated + Original", negated_splits + orig_splits)

    # Build table
    results_df = pd.DataFrame(rows)[
        ["split", "model", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
    ]

    # Pretty print and capture as string
    with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
        table_str = results_df.to_string(index=False)
        print(table_str)

    # Save to text file (metrics table as pretty table)
    results_path = "evaluation/results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(table_str)
        f.write("\n")

    # Append formatted confusion matrices to results file
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nConfusion matrices (rows=true, cols=pred):\n")
        for item in confusion_outputs:
            f.write(f"\n[{item['split']}] - {item['model']}\n")
            f.write(_format_confusion_matrix(item["cm"], item["labels"]))
            f.write("\n")
        f.write("\nLabel distributions (lowercased):\n")
        for s in label_summaries:
            f.write(f"\n[{s['split']}] n={s['n']}\n")
            # Order in standard NLI order if present
            order = ["entailment", "neutral", "contradiction"]
            counts = s["counts"]
            for k in order:
                if k in counts:
                    f.write(f"  {k}: {counts[k]}\n")
            # Any extra labels
            for k, v in counts.items():
                if k not in order:
                    f.write(f"  {k}: {v}\n")

    print(f"Saved results to {results_path}")
