from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .logger_config import setup_logger

logger = setup_logger()

class HFSequenceClassifier:
    """HF sequence classifier wrapper returning label + probabilities."""

    def __init__(self, model_id: str, device: int = None):
        logger.info("Initializing HF Sequence Classifier: %s", model_id)
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if self.device >= 0:
            self.model.to(torch.device(f"cuda:{self.device}"))
        self.model.eval()
        self.id2label = getattr(self.model.config, "id2label", {i: str(i) for i in range(self.model.config.num_labels)})

    def predict_with_probs(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            if self.device >= 0:
                enc = {k: v.to(torch.device(f"cuda:{self.device}")) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                idx = int(row.argmax())
                results.append({"label": self.id2label[idx], "probabilities": {self.id2label[i]: float(row[i]) for i in range(len(row))}})
        return results

class HFTokenClassifier:
    """NER / token-classifier wrapper (transformers pipeline)."""

    def __init__(self, model_id: str, device: int = None):
        logger.info("Initializing HF Token Classifier (NER): %s", model_id)
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        self.pipeline = pipeline("ner", model=model_id, tokenizer=model_id, device=self.device)

    def predict_ner(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        outs = []
        for i in range(0, len(texts), 16):
            chunk = texts[i:i+16]
            res = self.pipeline(chunk)
            if isinstance(res, list) and len(res) == len(chunk) and isinstance(res[0], list):
                outs.extend(res)
            else:
                outs.extend([res] if not isinstance(res, list) else res)
        return outs