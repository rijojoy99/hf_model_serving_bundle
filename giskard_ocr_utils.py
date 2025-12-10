from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
from typing import List
from .logger_config import setup_logger

logger = setup_logger()

def ocr_pdf_tesseract(pdf_path: str, dpi: int = 300) -> str:
    logger.info("OCR(pytesseract) extracting: %s", pdf_path)
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        logger.exception("pdf2image failed: %s", e)
        return f"<ocr_error: {e}>"
    texts: List[str] = []
    for p in pages:
        try:
            texts.append(pytesseract.image_to_string(p))
        except Exception as e:
            logger.exception("pytesseract page error: %s", e)
            texts.append(f"<ocr_page_error: {e}>")
    return "\n".join(texts)

class HFOCR:
    """Vision-encoder-decoder HF OCR (heavy; GPU recommended)."""

    def __init__(self, model_id: str):
        logger.info("Initializing HF OCR: %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.eval()
        if hasattr(self.model, "to") and __import__("torch").cuda.is_available():
            self.model.to("cuda")

    def image_to_text(self, image: Image.Image) -> str:
        try:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            if __import__("torch").cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            with __import__("torch").no_grad():
                outs = self.model.generate(pixel_values, max_length=512)
            return self.processor.batch_decode(outs, skip_special_tokens=True)[0]
        except Exception as e:
            logger.exception("HF OCR failed: %s", e)
            return f"<hf_ocr_error: {e}>"

    def pdf_to_text(self, pdf_path: str, dpi: int = 300) -> str:
        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            logger.exception("pdf2image failed: %s", e)
            return f"<ocr_error: {e}>"
        texts = [self.image_to_text(p) for p in pages]
        return "\n".join(texts)