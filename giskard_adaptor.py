from typing import Any, Dict, List
import json
from giskard import Model as GiskardModel, scan
from .logger_config import setup_logger

logger = setup_logger()

def build_giskard_model(kind: str, model_id: str,
                        vllm=None, seq_clf=None, token_clf=None, hf_ocr=None):
    kind = kind.lower()
    if kind == "generation":
        if vllm is None:
            raise ValueError("vllm required for generation")
        def fn(df):
            prompts = df["prompt"].astype(str).tolist()
            return vllm.generate_batch(prompts)
        return GiskardModel(model=fn, model_type="text_generation", name=f"vllm_gen:{model_id}", feature_names=["prompt"])

    if kind == "classifier":
        if seq_clf is None:
            raise ValueError("seq_clf required for classifier")
        def fn(df):
            texts = df["text"].astype(str).tolist()
            preds = seq_clf.predict_with_probs(texts)
            return [p["label"] for p in preds]
        return GiskardModel(model=fn, model_type="classifier", name=f"hf_seq:{model_id}", feature_names=["text"])

    if kind == "vllm_classifier":
        if vllm is None:
            raise ValueError("vllm required")
        def fn(df):
            texts = df["text"].astype(str).tolist()
            return vllm.classify_prompt(texts)
        return GiskardModel(model=fn, model_type="classifier", name=f"vllm_clf:{model_id}", feature_names=["text"])

    if kind == "token":
        if token_clf is None:
            raise ValueError("token_clf required")
        import json as _json
        def fn(df):
            texts = df["text"].astype(str).tolist()
            ner = token_clf.predict_ner(texts)
            return [_json.dumps(e) for e in ner]
        return GiskardModel(model=fn, model_type="other", name=f"hf_token:{model_id}", feature_names=["text"])

    if kind == "ocr":
        def fn(df):
            paths = df["pdf_path"].astype(str).tolist()
            out = []
            for p in paths:
                if hf_ocr is not None:
                    out.append(hf_ocr.pdf_to_text(p))
                else:
                    from .ocr_utils import ocr_pdf_tesseract
                    out.append(ocr_pdf_tesseract(p))
            return out
        return GiskardModel(model=fn, model_type="other", name=f"ocr:{model_id}", feature_names=["pdf_path"])

    if kind == "ocr_classifier":
        if seq_clf is None and vllm is None:
            raise ValueError("Provide seq_clf or vllm for ocr_classifier")
        def fn(df):
            paths = df["pdf_path"].astype(str).tolist()
            texts = []
            for p in paths:
                if hf_ocr is not None:
                    texts.append(hf_ocr.pdf_to_text(p))
                else:
                    from .ocr_utils import ocr_pdf_tesseract
                    texts.append(ocr_pdf_tesseract(p))
            if seq_clf:
                preds = seq_clf.predict_with_probs(texts)
                return [p["label"] for p in preds]
            else:
                return vllm.classify_prompt(texts)
        return GiskardModel(model=fn, model_type="classifier", name=f"ocr_clf:{model_id}", feature_names=["pdf_path"])

    raise ValueError(f"Unknown kind: {kind}")

# LLM judge utilities (short)
def extract_scan_issues(scan_res: Any, max_items: int = 10) -> List[Dict]:
    issues = []
    try:
        report = scan_res.to_dict() if hasattr(scan_res, "to_dict") else (scan_res if isinstance(scan_res, dict) else {"report": str(scan_res)})
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
            for i, it in enumerate(node):
                walk(it, f"{path}[{i}]")
    walk(report)
    # dedupe & return top
    uniq = {(i["test_name"], i["summary"]): i for i in issues}
    issues = list(uniq.values())[:max_items]
    return issues

def llm_judge(vllm, scan_res, top_k=5):
    issues = extract_scan_issues(scan_res, max_items=top_k)
    brief = "\n".join([f"- {i['test_name']}: {i['summary'][:200]} (severity={i['severity']})" for i in issues]) or "No issues found."
    prompt = (
        "You are an ML QA judge. Return ONLY valid JSON with keys: verdict(pass|fail|review), summary, actions(list).\n\n"
        f"Scan issues:\n{brief}\n\nReturn JSON."
    )
    out = vllm.generate_batch([prompt])[0] if vllm else "<no_llm>"
    judgement = {"raw": out, "issues": issues}
    try:
        first = out.find("{"); last = out.rfind("}")
        if first != -1 and last != -1:
            parsed = json.loads(out[first:last+1])
            judgement["parsed"] = parsed
            judgement["verdict"] = parsed.get("verdict")
            judgement["summary"] = parsed.get("summary")
            judgement["actions"] = parsed.get("actions")
        else:
            judgement["verdict"] = "review" if issues else "pass"
            judgement["summary"] = out.strip()[:500]
            judgement["actions"] = []
    except Exception as e:
        logger.exception("Failed parsing judge output: %s", e)
        judgement.update({"parse_error": str(e), "verdict": "review"})
    logger.info("Judge verdict: %s", judgement.get("verdict"))
    return judgement