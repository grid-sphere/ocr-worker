
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests
import runpod
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aicos-ocr")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
STORAGE_HOST = os.environ.get("STORAGE_HOST", "").lower().strip()
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 100

logger.info("Loading DeepSeek-OCR model...")
if not torch.cuda.is_available():
    raise RuntimeError("DeepSeek-OCR requires a CUDA GPU worker.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

logger.info("DeepSeek-OCR loaded successfully.")


def download_file(file_url, destination):
    parsed = urlparse(file_url)

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("file_url must be an HTTPS URL.")

    if not STORAGE_HOST or parsed.hostname.lower() != STORAGE_HOST:
        raise ValueError("File URL host is not in the configured allowlist.")

    total = 0
    with requests.get(
        file_url,
        stream=True,
        timeout=(15, 120),
        allow_redirects=False,
    ) as response:
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 25 MB limit.")

        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError("File exceeds the 25 MB limit.")

                output.write(chunk)

    if total == 0:
        raise ValueError("Downloaded file is empty.")


def infer_page(image_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    result = model.infer(
        tokenizer,
        prompt="<image>\nFree OCR.",
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
    )

    # DeepSeek-OCR versions may return text directly or write result.mmd.
    markdown_file = output_dir / "result.mmd"
    if markdown_file.exists():
        return markdown_file.read_text(encoding="utf-8").strip()

    if isinstance(result, str):
        return result.strip()

    return str(result or "").strip()


def handler(event):
    job_input = event.get("input", {})
    file_url = job_input.get("file_url")
    filename = job_input.get("filename", "submission.pdf")

    if not file_url:
        return {"success": False, "error": "Missing file_url."}

    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        return {"success": False, "error": "Unsupported file type."}

    try:
        with tempfile.TemporaryDirectory(prefix="aicos-ocr-") as temp:
            work_dir = Path(temp)
            input_path = work_dir / f"input{extension}"
            download_file(file_url, input_path)

            page_images = []

            if extension == ".pdf":
                with fitz.open(input_path) as pdf:
                    if len(pdf) == 0 or len(pdf) > MAX_PDF_PAGES:
                        raise ValueError(
                            f"PDF must contain 1 to {MAX_PDF_PAGES} pages."
                        )

                    for index, page in enumerate(pdf):
                        image_path = work_dir / f"page_{index + 1}.png"
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False,
                        )
                        pix.save(str(image_path))
                        page_images.append(image_path)
            else:
                page_images.append(input_path)

            pages = []
            all_text = []

            for index, image_path in enumerate(page_images, start=1):
                logger.info("Processing page %s of %s", index, len(page_images))

                text = infer_page(
                    image_path,
                    work_dir / f"output_{index}",
                )

                pages.append({
                    "page_number": index,
                    "text": text,
                })

                if text:
                    all_text.append(f"## Page {index}\n\n{text}")

            combined_text = "\n\n".join(all_text)

            return {
                "success": bool(combined_text.strip()),
                "filename": Path(filename).name,
                "document_type": extension.lstrip("."),
                "text": combined_text,
                "pages": pages,
                "model": MODEL_NAME,
            }

    except Exception:
        logger.exception("OCR job failed")
        raise


runpod.serverless.start({"handler": handler})
