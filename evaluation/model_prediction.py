from huggingface_hub import login
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


class ModelPrediction:
    def __init__(self, model_id_1, model_id_2, max_length: int = 128):
        hf_token = os.environ.get("HF_TOKEN_SD")
        if hf_token:
            login(token=hf_token)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer_base = AutoTokenizer.from_pretrained(model_id_1)
        self.config_base = AutoConfig.from_pretrained(model_id_1)
        self.model_base = (
            AutoModelForSequenceClassification.from_pretrained(model_id_1)
            .to(self.device)
            .eval()
        )

        self.tokenizer_debiased = AutoTokenizer.from_pretrained(model_id_2)
        self.config_debiased = AutoConfig.from_pretrained(model_id_2)
        self.model_debiased = (
            AutoModelForSequenceClassification.from_pretrained(model_id_2)
            .to(self.device)
            .eval()
        )

        # Tokenizer configuration
        self.max_length = int(max_length) if max_length is not None else 128

    def tokenize_input(self, premise, hypothesis):
        input_base = self.tokenizer_base(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_debiased = self.tokenizer_debiased(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return input_base, input_debiased

    def predict(self, input_base, input_debiased):
        with torch.no_grad():
            input_base = {k: v.to(self.device) for k, v in input_base.items()}
            input_debiased = {k: v.to(self.device) for k, v in input_debiased.items()}

            outputs_base = self.model_base(**input_base)
            outputs_debiased = self.model_debiased(**input_debiased)

            logits_base = outputs_base.logits
            logits_debiased = outputs_debiased.logits

            pred_base_id = logits_base.argmax(dim=-1).item()
            pred_debiased_id = logits_debiased.argmax(dim=-1).item()

            probs_base = (
                F.softmax(logits_base, dim=-1).squeeze(0).detach().cpu().tolist()
            )
            probs_debiased = (
                F.softmax(logits_debiased, dim=-1).squeeze(0).detach().cpu().tolist()
            )
            logits_base_list = logits_base.squeeze(0).detach().cpu().tolist()
            logits_debiased_list = logits_debiased.squeeze(0).detach().cpu().tolist()

        # Map to string labels using model configs
        pred_base = self.config_base.id2label.get(pred_base_id, str(pred_base_id))
        pred_debiased = self.config_debiased.id2label.get(
            pred_debiased_id, str(pred_debiased_id)
        )
        return (
            pred_base.upper(),
            pred_debiased.upper(),
            probs_base,
            probs_debiased,
            logits_base_list,
            logits_debiased_list,
        )
