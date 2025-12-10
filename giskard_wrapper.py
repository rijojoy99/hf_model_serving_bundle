from typing import List, Optional
from vllm import LLM, SamplingParams
from .logger_config import setup_logger

logger = setup_logger()

class VLLMWrapper:
    """Wrapper around vLLM for HF LLMs — batched generation, chat, prompt-based classification."""

    def __init__(self, model_id: str, sampling: Optional[SamplingParams] = None):
        self.model_id = model_id
        logger.info("Initializing vLLM model: %s", model_id)
        self.llm = LLM(model=model_id)
        self.sampling = sampling or SamplingParams(temperature=0.0, max_tokens=256)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        logger.debug("vLLM.generate_batch: n=%d", len(prompts))
        outputs = self.llm.generate(prompts, self.sampling)
        results = []
        for out in outputs:
            results.append(out.outputs[0].text if out.outputs else "")
        return results

    def chat_batch(self, conversations: List[List[dict]]) -> List[str]:
        logger.debug("vLLM.chat_batch: n=%d", len(conversations))
        outputs = self.llm.chat(conversations, self.sampling)
        results = [o.outputs[0].text if o.outputs else "" for o in outputs]
        return results

    def classify_prompt(self, texts: List[str], label_set: Optional[List[str]] = None, batch_size: int = 16) -> List[str]:
        def template(t):
            hint = f" Allowed labels: {', '.join(label_set)}." if label_set else ""
            return f"Task: Classify the following text.{hint}\n\nText:\n\"\"\"\n{t}\n\"\"\"\n\nRespond with a single label (one word)."

        prompts = [template(t) for t in texts]
        preds = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i+batch_size]
            outs = self.generate_batch(chunk)
            for o in outs:
                s = (o or "").strip().splitlines()[0].strip()
                token = s.split()[0] if s else "UNKNOWN"
                preds.append(token.upper())
        return preds