pip install vllm==0.5.4 transformers==4.40.2 torch==2.2.2 accelerate==0.30.1 numpy==1.26.4 pandas==2.2.1 pydantic==1.10.15 starlette==0.27.0 giskard==2.0.0 pdf2image==1.17.0 pytesseract==0.3.10 pillow==10.2.0


pip install \
    vllm==0.5.4 \
    transformers==4.40.2 \
    torch==2.2.2 \
    accelerate==0.30.1 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    pydantic==1.10.15 \
    starlette==0.27.0 \
    giskard==2.0.0 \
    pdf2image==1.17.0 \
    pytesseract==0.3.10 \
    pillow==10.2.0


#!/usr/bin/env python3
"""
vllm_giskard_runner.py

Usage examples:
  # run a vLLM generation model scan (default small models)
  python vllm_giskard_runner.py --model-id facebook/opt-125m --model-kind generation

  # run a HF sequence-classifier scan
  python vllm_giskard_runner.py --model-id distilbert-base-uncased-finetuned-sst-2-english --model-kind classifier

  # run OCR extraction + classifier scan (use local PDF paths)
  python vllm_giskard_runner.py --model-id distilbert-base-uncased-finetuned-sst-2-english --model-kind ocr_classifier --pdf-list sample_pdfs.txt

  # use HF OCR model (heavy)
  export USE_HF_OCR=true
  python vllm_giskard_runner.py --model-id distilbert-base-uncased-finetuned-sst-2-english --model-kind ocr_classifier --pdf-list sample_pdfs.txt --hf-ocr-model microsoft/trocr-base-printed
"""

import os
import argparse
import json
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict
import pandas as pd
import traceback

# vLLM
from vllm import LLM, SamplingParams

# transformers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
# optional HF OCR imports (only used if requested)
from transformers import AutoProcessor, VisionEncoderDecoderModel

# OCR utilities
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Giskard
from giskard import Model as GiskardModel, scan

# -----------------------
# Logging
# -----------------------
LOG_DIR = os.getenv("VLLM_LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"vllm_giskard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("vllm_giskard")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
logger.info("Logger started; log file: %s", LOG_FILE)

# -----------------------
# Utilities
# -----------------------
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = []
        for _ in range(size):
            try:
                chunk.append(next(it))
            except StopIteration:
                break
        if not chunk:
            break
        yield chunk

# -----------------------
# vLLM Wrapper (HF models)
# -----------------------
class VLLMHF:
    def __init__(self, model_id: str, sampling: Optional[SamplingParams] = None):
        logger.info("Initializing vLLM model: %s", model_id)
        try:
            self.llm = LLM(model=model_id)
        except Exception as e:
            logger.exception("Failed to init vLLM: %s", e)
            raise
        self.sampling = sampling or SamplingParams(temperature=0.0, max_tokens=256)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        if len(prompts) == 0:
            return []
        logger.debug("vLLM.generate_batch inputs=%d", len(prompts))
        try:
            outputs = self.llm.generate(prompts, self.sampling)
        except Exception as e:
            logger.exception("vLLM.generate failed: %s", e)
            return [f"<llm_error: {e}>" for _ in prompts]
        res = []
        for r in outputs:
            res.append(r.outputs[0].text if r.outputs else "")
        return res

    def classify_prompt_based(self, texts: List[str], label_set: Optional[List[str]] = None, batch_size: int = 16) -> List[str]:
        def template(t):
            hint = f" Allowed labels: {', '.join(label_set)}." if label_set else ""
            return (
                f"Task: Classify the following text.{hint}\n\n"
                f"Text:\n\"\"\"\n{t}\n\"\"\"\n\nRespond with a single label (one word)."
            )
        prompts = [template(t) for t in texts]
        preds = []
        for chunk in chunked(prompts, batch_size):
            outs = self.generate_batch(chunk)
            for o in outs:
                s = (o or "").strip().splitlines()[0].strip()
                token = s.split()[0] if s else "UNKNOWN"
                preds.append(token.upper())
        return preds

# -----------------------
# HF sequence classifier wrapper
# -----------------------
class HFSequenceClassifier:
    def __init__(self, model_id: str, device: Optional[int] = None):
        logger.info("Initializing HF sequence-classifier: %s", model_id)
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
        results = []
        for chunk in chunked(texts, batch_size):
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

# -----------------------
# HF Token classifier / NER (pipeline wrapper)
# -----------------------
class HFTokenClassifier:
    def __init__(self, model_id: str, device: Optional[int] = None):
        logger.info("Initializing HF token-classifier (NER) model: %s", model_id)
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        self.pipeline = hf_pipeline("ner", model=model_id, tokenizer=model_id, device=self.device)

    def predict_ner(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        outs = []
        for chunk in chunked(texts, 16):
            res = self.pipeline(chunk)
            # pipeline returns list-of-lists when input is list
            if isinstance(res, list) and len(res) == len(chunk) and isinstance(res[0], list):
                outs.extend(res)
            else:
                outs.extend([res] if not isinstance(res, list) else res)
        return outs

# -----------------------
# OCR helpers (pytesseract + optional HF OCR)
# -----------------------
def ocr_from_pdf_tesseract(pdf_path: str, dpi: int = 300) -> str:
    logger.info("OCR via pytesseract: %s", pdf_path)
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        logger.exception("pdf2image failed: %s", e)
        return f"<ocr_error: {e}>"
    texts = []
    for page in pages:
        try:
            texts.append(pytesseract.image_to_string(page))
        except Exception as e:
            logger.exception("tesseract page error: %s", e)
            texts.append(f"<ocr_page_error: {e}>")
    return "\n".join(texts)

class HFOCR:
    def __init__(self, model_id: str):
        logger.info("Initializing HF OCR model (vision-encoder-decoder): %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def image_to_text(self, image: Image.Image) -> str:
        try:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            with torch.no_grad():
                outs = self.model.generate(pixel_values, max_length=512)
            return self.processor.batch_decode(outs, skip_special_tokens=True)[0]
        except Exception as e:
            logger.exception("HF OCR generation failed: %s", e)
            return f"<hf_ocr_error: {e}>"

    def pdf_to_text(self, pdf_path: str, dpi: int = 300) -> str:
        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            logger.exception("pdf2image failed for HF OCR: %s", e)
            return f"<ocr_error: {e}>"
        out_texts = []
        for p in pages:
            out_texts.append(self.image_to_text(p))
        return "\n".join(out_texts)

# -----------------------
# LLM Judge utilities
# -----------------------
def extract_scan_issues(scan_res: Any, max_items: int = 10):
    issues = []
    try:
        if hasattr(scan_res, "to_dict"):
            report = scan_res.to_dict()
        elif isinstance(scan_res, dict):
            report = scan_res
        else:
            report = {"report": str(scan_res)}
    except Exception:
        report = {"report": str(scan_res)}

    def walk(node, path=""):
        if isinstance(node, dict):
            status = node.get("status") or node.get("result") or node.get("outcome")
            if isinstance(status, str) and status.lower() in ("failed", "warning", "error"):
                summary = node.get("message") or node.get("summary") or json.dumps(node)[:300]
                issues.append({"test_name": path or "root", "severity": status, "summary": summary})
            for k, v in node.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
    walk(report)
    # dedupe & sort
    def severity_score(s: str) -> int:
        s = (s or "").lower()
        if "failed" in s or "error" in s:
            return 3
        if "warning" in s:
            return 2
        return 1
    uniq = {(i['test_name'], i['summary']): i for i in issues}
    issues = list(uniq.values())
    issues.sort(key=lambda x: severity_score(x.get("severity","")), reverse=True)
    return issues[:max_items]

def llm_judge_scan(vllm: VLLMHF, scan_res: Any, top_k: int = 5):
    issues = extract_scan_issues(scan_res, max_items=top_k)
    if issues:
        brief = "\n".join([f"- {i['test_name']}: {i['summary'][:300]} (severity={i['severity']})" for i in issues])
    else:
        brief = "No failing or warning tests detected in the scan."

    prompt = (
        "You are an expert ML QA judge.\n"
        "Given the short list of issues found by an automated model scan tool, return a JSON object with keys:\n"
        "  - verdict: one of 'pass', 'fail', or 'review'\n"
        "  - summary: a 1-2 sentence summary of the most critical problems (or 'All good' when none)\n"
        "  - actions: a prioritized list (1-5) of concrete remediation actions (short sentences).\n"
        "Only output valid JSON and nothing else.\n\n"
        f"Scan issues:\n{brief}\n\nRespond with only a JSON object."
    )
    logger.debug("LLM judge prompt len=%d", len(prompt))
    try:
        llm_out = vllm.generate_batch([prompt])[0]
    except Exception as e:
        logger.exception("LLM judge failed to generate: %s", e)
        llm_out = f"<llm_error: {e}>"

    judgement = {"raw": llm_out}
    try:
        first = llm_out.find("{")
        last = llm_out.rfind("}")
        if first != -1 and last != -1 and last > first:
            parsed = json.loads(llm_out[first:last+1])
            judgement["parsed_json"] = parsed
            judgement["verdict"] = parsed.get("verdict") if isinstance(parsed.get("verdict"), str) else ("fail" if issues else "pass")
            judgement["summary"] = parsed.get("summary", "")
            judgement["actions"] = parsed.get("actions", [])
        else:
            judgement["verdict"] = "review" if issues else "pass"
            judgement["summary"] = llm_out.strip()[:500]
            judgement["actions"] = []
    except Exception as e:
        logger.exception("Failed to parse LLM judge output: %s", e)
        judgement.update({"parse_error": str(e), "verdict": "review", "summary": llm_out[:500], "actions": []})

    judgement["issues"] = issues
    logger.info("LLM judge verdict=%s", judgement.get("verdict"))
    return judgement

# -----------------------
# Giskard wrappers factory (parametrized)
# -----------------------
def build_giskard_model(model_kind: str, model_id: str, *,
                        vllm: Optional[VLLMHF] = None,
                        seq_clf: Optional[HFSequenceClassifier] = None,
                        token_clf: Optional[HFTokenClassifier] = None,
                        hf_ocr: Optional[HFOCR] = None,
                        logger_local = logger):
    """
    model_kind: one of:
      - 'generation' : use vLLM to generate text from 'prompt' column
      - 'classifier' : use HF sequence classifier on 'text' column
      - 'vllm_classifier' : prompt-based vLLM classifier on 'text' column
      - 'token' : NER token classifier on 'text' column (returns JSON-string per row)
      - 'ocr' : OCR extraction model (returns extracted text) from 'pdf_path' (pytesseract or HF OCR)
      - 'ocr_classifier' : OCR->sequence-classifier pipeline (pdf_path -> extracted text -> classifier)
    """
    model_kind = model_kind.lower()
    if model_kind == "generation":
        if vllm is None:
            raise ValueError("vllm instance required for generation")
        def fn(df):
            prompts = df["prompt"].astype(str).tolist()
            return vllm.generate_batch(prompts)
        return GiskardModel(model=fn, model_type="text_generation", name=f"vllm_gen:{model_id}", description=f"vLLM generation {model_id}", feature_names=["prompt"])

    if model_kind == "classifier":
        if seq_clf is None:
            raise ValueError("seq_clf required for classifier")
        def fn(df):
            texts = df["text"].astype(str).tolist()
            preds = seq_clf.predict_with_probs(texts)
            # Giskard often expects list[str]; return top-label strings
            return [p["label"] for p in preds]
        return GiskardModel(model=fn, model_type="classifier", name=f"hf_seq:{model_id}", description=f"HF seq classifier {model_id}", feature_names=["text"])

    if model_kind == "vllm_classifier":
        if vllm is None:
            raise ValueError("vllm required for vllm_classifier")
        def fn(df):
            texts = df["text"].astype(str).tolist()
            return vllm.classify_prompt_based(texts)
        return GiskardModel(model=fn, model_type="classifier", name=f"vllm_clf:{model_id}", description=f"vLLM prompt classifier {model_id}", feature_names=["text"])

    if model_kind == "token":
        if token_clf is None:
            raise ValueError("token_clf required for token")
        import json as _json
        def fn(df):
            texts = df["text"].astype(str).tolist()
            ner_lists = token_clf.predict_ner(texts)
            # return JSON string per row
            return [_json.dumps(ents) for ents in ner_lists]
        return GiskardModel(model=fn, model_type="other", name=f"hf_token:{model_id}", description=f"HF token classifier {model_id}", feature_names=["text"])

    if model_kind == "ocr":
        # returns extracted text from pdf_path
        def fn(df):
            paths = df["pdf_path"].astype(str).tolist()
            out = []
            for p in paths:
                if hf_ocr is not None:
                    out.append(hf_ocr.pdf_to_text(p))
                else:
                    out.append(ocr_from_pdf_tesseract(p))
            return out
        return GiskardModel(model=fn, model_type="other", name=f"ocr_extractor:{model_id}", description=f"OCR extractor {model_id}", feature_names=["pdf_path"])

    if model_kind == "ocr_classifier":
        # pipeline: pdf_path -> extracted text -> classifier (seq_clf or vllm)
        if seq_clf is None and vllm is None:
            raise ValueError("Either seq_clf or vllm must be provided for ocr_classifier")
        def fn(df):
            paths = df["pdf_path"].astype(str).tolist()
            texts = []
            for p in paths:
                if hf_ocr is not None:
                    texts.append(hf_ocr.pdf_to_text(p))
                else:
                    texts.append(ocr_from_pdf_tesseract(p))
            if seq_clf is not None:
                preds = seq_clf.predict_with_probs(texts)
                return [p["label"] for p in preds]
            else:
                return vllm.classify_prompt_based(texts)
        return GiskardModel(model=fn, model_type="classifier", name=f"ocr_pipeline:{model_id}", description=f"OCR->classifier pipeline {model_id}", feature_names=["pdf_path"])

    raise ValueError(f"Unknown model_kind: {model_kind}")

# -----------------------
# CLI / main flow
# -----------------------
def load_pdf_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="Hugging Face model id or local path")
    p.add_argument("--model-kind", required=True, choices=["generation","classifier","vllm_classifier","token","ocr","ocr_classifier"], help="Kind of model/wrapper to run and scan")
    p.add_argument("--hf-ocr-model", default=os.getenv("HF_OCR_MODEL"), help="Optional HF OCR model id (vision-encoder-decoder) to use instead of pytesseract")
    p.add_argument("--pdf-list", help="Path to newline-separated list of PDF file paths (for OCR flows)")
    p.add_argument("--output-prefix", default="giskard_demo", help="prefix for saved artifacts (report / judgement)")
    args = p.parse_args()

    model_id = args.model_id
    model_kind = args.model_kind
    hf_ocr_model = args.hf_ocr_model
    pdf_list = args.pdf_list
    output_prefix = args.output_prefix

    # instantiate runtimes as required
    vllm = None
    seq_clf = None
    token_clf = None
    hf_ocr = None

    try:
        if model_kind in ("generation", "vllm_classifier", "ocr_classifier"):
            logger.info("Initializing vLLM for model-kind=%s", model_kind)
            vllm = VLLMHF(model_id=model_id)

        if model_kind in ("classifier", "ocr_classifier"):
            # For classifier flows, model_id should be seq-classifier id; allow overriding if user passed different id.
            # If user wants to use different HF seq classifier, they can pass its id via --model-id.
            logger.info("Initializing HF sequence-classifier for model-id=%s", model_id)
            seq_clf = HFSequenceClassifier(model_id=model_id)

        if model_kind == "token":
            logger.info("Initializing HF token-classifier for model-id=%s", model_id)
            token_clf = HFTokenClassifier(model_id=model_id)

        if model_kind in ("ocr","ocr_classifier"):
            # load HF OCR if requested via flag / env
            if hf_ocr_model:
                try:
                    hf_ocr = HFOCR(model_id=hf_ocr_model)
                except Exception:
                    logger.exception("HF OCR init failed; will fallback to pytesseract")
                    hf_ocr = None
            else:
                logger.info("No HF OCR model id provided; using pytesseract by default")

    except Exception as e:
        logger.exception("Initialization failed: %s", e)
        raise

    # prepare test dataset(s)
    if model_kind in ("generation",):
        df = pd.DataFrame({"prompt": ["Write a 2-line poem about coffee.", "Summ