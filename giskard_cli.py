import argparse
import os
import json
import pandas as pd
from .logger_config import setup_logger
from .vllm_wrapper import VLLMWrapper
from .hf_classifiers import HFSequenceClassifier, HFTokenClassifier
from .ocr_utils import HFOCR
from .giskard_adapter import build_giskard_model, llm_judge
from giskard import scan

logger = setup_logger()

def load_pdf_list(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-kind", required=True, choices=["generation","classifier","vllm_classifier","token","ocr","ocr_classifier"])
    p.add_argument("--hf-ocr-model", default=os.getenv("HF_OCR_MODEL"))
    p.add_argument("--pdf-list")
    p.add_argument("--output-prefix", default="giskard_demo")
    args = p.parse_args()

    model_id = args.model_id
    kind = args.model_kind
    hf_ocr_model = args.hf_ocr_model
    pdf_list = args.pdf_list
    prefix = args.output_prefix

    # instantiate runtimes as required
    vllm = None; seq_clf = None; token_clf = None; hf_ocr = None
    if kind in ("generation","vllm_classifier","ocr_classifier"):
        vllm = VLLMWrapper(model_id=model_id)
    if kind in ("classifier","ocr_classifier"):
        seq_clf = HFSequenceClassifier(model_id=model_id)
    if kind == "token":
        token_clf = HFTokenClassifier(model_id=model_id)
    if kind in ("ocr","ocr_classifier") and hf_ocr_model:
        try:
            hf_ocr = HFOCR(hf_ocr_model)
        except Exception:
            logger.exception("HF OCR init failed; will use pytesseract fallback")
            hf_ocr = None

    if kind == "ocr" or kind == "ocr_classifier":
        pdfs = load_pdf_list(pdf_list) if pdf_list else []
        df = pd.DataFrame({"pdf_path": pdfs})
    elif kind == "generation":
        df = pd.DataFrame({"prompt": ["Write a two-line poem about coffee.", "Summarize: Product launch went well."]})
    else:
        df = pd.DataFrame({"text": ["I loved the product, arrived on time.", "Terrible service and rude staff."]})

    # build giskard model
    gmodel = build_giskard_model(kind, model_id, vllm=vllm, seq_clf=seq_clf, token_clf=token_clf, hf_ocr=hf_ocr)

    # run scan
    logger.info("Running Giskard scan for kind=%s id=%s rows=%d", kind, model_id, len(df))
    try:
        scan_res = scan(gmodel, df)
        if hasattr(scan_res, "to_html"):
            html = f"{prefix}_scan.html"
            scan_res.to_html(html)
            logger.info("Saved HTML report: %s", html)
        elif hasattr(scan_res, "to_dict"):
            js = f"{prefix}_scan.json"
            with open(js, "w") as f:
                json.dump(scan_res.to_dict(), f, indent=2)
            logger.info("Saved JSON report: %s", js)
        else:
            with open(f"{prefix}_scan.json", "w") as f:
                json.dump({"scan": str(scan_res)}, f, indent=2)
            logger.info("Saved fallback scan result")
    except Exception as e:
        logger.exception("Giskard scan failed: %s", e)
        scan_res = {"error": str(e)}

    # LLM judge if available
    if vllm:
        judge = llm_judge(vllm, scan_res)
        outj = f"{prefix}_judgement.json"
        with open(outj, "w") as f:
            json.dump(judge, f, indent=2)
        logger.info("Saved judgement: %s", outj)
        print(json.dumps(judge, indent=2))
    else:
        logger.info("No vLLM available; skipping judge")

if __name__ == "__main__":
    main()