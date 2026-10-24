import logging
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests
import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek-ocr")

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "deepseek-ai/DeepSeek-OCR",
)

# Set this to the hostname of your private object storage.
# Example: storage.example.com
# Multiple hosts may be comma-separated.
STORAGE_HOSTS = {
    host.strip().lower()
    for host in os.getenv("STORAGE_HOSTS", "").split(",")
    if host.strip()
}

MAX_FILES_PER_JOB = int(os.getenv("MAX_FILES_PER_JOB", "50"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
DOWNLOAD_TIMEOUT = (10, 60)

OCR_PROMPT = "<image>\nFree OCR."

if not STORAGE_HOSTS:
    logger.warning(
        "STORAGE_HOSTS is not configured. File downloads will be rejected."
    )


# ---------------------------------------------------------
# Load model once per worker
# ---------------------------------------------------------

logger.info("Loading tokenizer: %s", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model_kwargs = {
    "trust_remote_code": True,
    "use_safetensors": True,
    "device_map": "auto",
}

if torch.cuda.is_available():
    model_kwargs["torch_dtype"] = torch.bfloat16
else:
    model_kwargs["torch_dtype"] = torch.float32

logger.info("Loading model: %s", MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    **model_kwargs,
).eval()

logger.info("DeepSeek-OCR model loaded successfully.")


# ---------------------------------------------------------
# Secure file downloading
# ---------------------------------------------------------

def validate_download_url(url):
    if not isinstance(url, str) or not url:
        raise ValueError("file_url is required.")

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError("Only HTTPS download URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("Invalid download URL.")

    if parsed.username or parsed.password:
        raise ValueError("Credentials embedded in URLs are not allowed.")

    hostname = parsed.hostname.lower()

    if hostname not in STORAGE_HOSTS:
        raise ValueError(
            f"Download host is not allowed: {hostname}"
        )

    if parsed.port not in (None, 443):
        raise ValueError("Non-standard HTTPS ports are not allowed.")

    return url


def download_file(url, destination):
    url = validate_download_url(url)

    # Do not follow redirects. This prevents redirect-based host bypasses.
    with requests.get(
        url,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
        allow_redirects=False,
    ) as response:
        if response.is_redirect:
            raise ValueError(
                "Storage URL redirected. Supply a direct download URL."
            )

        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length:
            if int(content_length) > MAX_FILE_BYTES:
                raise ValueError("File exceeds the configured size limit.")

        total_bytes = 0

        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_BYTES:
                    raise ValueError(
                        "File exceeds the configured size limit."
                    )

                output.write(chunk)

    if total_bytes == 0:
        raise ValueError("Downloaded file is empty.")

    return total_bytes


# ---------------------------------------------------------
# OCR inference
# ---------------------------------------------------------

def infer_image(image_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    result = model.infer(
        tokenizer,
        prompt=OCR_PROMPT,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
    )

    # DeepSeek-OCR versions may return a result directly or write
    # the OCR output to a generated file. Check both possibilities.
    if isinstance(result, str) and result.strip():
        return result.strip()

    if isinstance(result, dict):
        for key in ("text", "markdown", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    candidates = sorted(
        output_dir.rglob("*.mmd"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        return candidates[0].read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    # Some versions save text under another extension.
    candidates = sorted(
        [
            *output_dir.rglob("*.txt"),
            *output_dir.rglob("*.md"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        return candidates[0].read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    raise RuntimeError(
        "DeepSeek-OCR completed without a recognizable text result. "
        "Check the model's generated output files and API."
    )


def process_one_file(file_info, job_dir):
    if not isinstance(file_info, dict):
        raise ValueError("Each file entry must be an object.")

    file_url = file_info.get("file_url")
    filename = Path(
        str(file_info.get("filename") or "document.bin")
    ).name

    try:
        page_number = int(file_info.get("page_number", 1))
    except (ValueError, TypeError):
        raise ValueError("page_number must be an integer.")

    if page_number < 1:
        raise ValueError("page_number must be positive.")

    extension = Path(filename).suffix.lower()

    if extension not in {
        ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
    }:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")

    source_path = job_dir / f"source_{uuid.uuid4().hex}{extension}"

    size = download_file(file_url, source_path)
    logger.info("Downloaded %s (%s bytes)", filename, size)

    page_results = []

    if extension == ".pdf":
        with fitz.open(source_path) as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")

            if len(pdf) == 0:
                raise ValueError("PDF contains no pages.")

            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError(
                    f"PDF exceeds the {MAX_PDF_PAGES}-page limit."
                )

            for index, pdf_page in enumerate(pdf):
                image_path = job_dir / (
                    f"pdf_{uuid.uuid4().hex}_{index + 1}.png"
                )

                pixmap = pdf_page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )
                pixmap.save(str(image_path))

                text = infer_image(
                    image_path,
                    job_dir / f"output_{uuid.uuid4().hex}",
                )

                page_results.append({
                    "page_number": page_number,
                    "source_page_number": index + 1,
                    "filename": filename,
                    "text": text,
                })

                image_path.unlink(missing_ok=True)

    else:
        # Verify the downloaded file is actually a readable image.
        with Image.open(source_path) as image:
            image.verify()

        text = infer_image(
            source_path,
            job_dir / f"output_{uuid.uuid4().hex}",
        )

        page_results.append({
            "page_number": page_number,
            "source_page_number": 1,
            "filename": filename,
            "text": text,
        })

    source_path.unlink(missing_ok=True)

    return page_results


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(event):
    payload = event.get("input", {})

    if not isinstance(payload, dict):
        return {
            "success": False,
            "error": "input must be a JSON object.",
        }

    files = payload.get("files")

    # Optional compatibility for testing a single file.
    if files is None and payload.get("file_url"):
        files = [{
            "file_url": payload["file_url"],
            "filename": payload.get("filename", "document.pdf"),
            "page_number": payload.get("page_number", 1),
        }]

    if not isinstance(files, list) or not files:
        return {
            "success": False,
            "error": "Provide a non-empty input.files list.",
        }

    if len(files) > MAX_FILES_PER_JOB:
        return {
            "success": False,
            "error": f"Maximum {MAX_FILES_PER_JOB} files per job.",
        }

    all_pages = []
    file_errors = []

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as temp:
        job_dir = Path(temp)

        for index, file_info in enumerate(files):
            try:
                results = process_one_file(file_info, job_dir)
                all_pages.extend(results)

            except Exception as exc:
                logger.exception("OCR failed for file index %s", index)

                info = file_info if isinstance(file_info, dict) else {}
                file_errors.append({
                    "index": index,
                    "page_number": info.get("page_number"),
                    "filename": Path(
                        str(info.get("filename") or "unknown")
                    ).name,
                    "error": str(exc)[:500],
                })

    # Keep successful pages even when a different file fails.
    all_pages.sort(
        key=lambda page: (
            page["page_number"],
            page["source_page_number"],
        )
    )

    combined_text = "\n\n".join(
        f"## Page {page['page_number']}"
        + (
            f" — PDF page {page['source_page_number']}"
            if page["source_page_number"] != 1
            else ""
        )
        + f"\n\n{page['text']}"
        for page in all_pages
    )

    return {
        "success": len(all_pages) > 0 and len(file_errors) == 0,
        "partial_success": len(all_pages) > 0 and len(file_errors) > 0,
        "text": combined_text,
        "pages": all_pages,
        "errors": file_errors,
        "model": MODEL_NAME,
    }


# This line is required by RunPod Serverless.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
