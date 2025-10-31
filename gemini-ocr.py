#!/usr/bin/env python3
import argparse
import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
# from http.client import responses
from typing import List, Any, Dict, Optional
from zoneinfo import ZoneInfo

# import argcomplete
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# TODO: save prompt to root of input dir?
# TODO: save temperature (any other confif info changes?) to a comment in the hOCR output file?

# Load environment variables from .env_gemini if present. This allows a project-local
# API key file while still supporting environment variables in other environments.
load_dotenv(dotenv_path=".env_gemini")


# ============================================================================
# QUOTA TRACKING SYSTEM
# ============================================================================


class QuotaTracker:
    """Tracks API usage quotas for different LLM models."""

    def __init__(self, config_path: str = "gemini-quota-config.json"):
        self.config_path = config_path
        self.config: Dict = {}
        self.usage: Dict[str, Dict] = defaultdict(
            lambda: {
                "requests_per_minute": [],
                "requests_per_day": [],
                "tokens_per_minute": [],
                "tokens_per_day": [],
            }
        )
        self.config_loaded = False
        self.warned_models = set()  # Track which models we've warned about

        self._load_config()

    def _load_config(self):
        """Load quota configuration from JSON file."""
        if not os.path.exists(self.config_path):
            logging.warning(
                f"Quota config file not found: {self.config_path}. "
                "Quota tracking will be disabled."
            )
            return

        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            self.config_loaded = True
            logging.debug(f"Loaded quota config from {self.config_path}")
        except Exception as e:
            logging.warning(
                f"Failed to load quota config from {self.config_path}: {e}. "
                "Quota tracking will be disabled."
            )

    def _get_reset_time(self, model: str, quota_type: str) -> Optional[datetime]:
        """Get the next reset time for a daily quota."""
        if not self.config_loaded or model not in self.config.get("models", {}):
            return None

        model_config = self.config["models"][model]
        quota_config = model_config.get("quotas", {}).get(quota_type, {})

        if "reset_time" not in quota_config:
            return None

        reset_time_str = quota_config["reset_time"]
        timezone_str = quota_config.get("reset_timezone", "UTC")

        try:
            # Parse reset time (HH:MM format)
            reset_hour, reset_minute = map(int, reset_time_str.split(":"))
            tz = ZoneInfo(timezone_str)

            # Get current time in the target timezone
            now = datetime.now(tz)

            # Create today's reset time
            reset_today = now.replace(
                hour=reset_hour, minute=reset_minute, second=0, microsecond=0
            )

            # If reset time has passed today, use tomorrow's reset time
            if now >= reset_today:
                from datetime import timedelta

                reset_next = reset_today + timedelta(days=1)
            else:
                reset_next = reset_today

            return reset_next
        except Exception as e:
            logging.debug(f"Failed to parse reset time for {model}: {e}")
            return None

    # TODO: get current quota usage from web

    def _clean_old_entries(self, model: str):
        """Remove entries older than their tracking window."""
        now = time.time()

        # Clean per-minute entries (keep last 60 seconds)
        self.usage[model]["requests_per_minute"] = [
            t for t in self.usage[model]["requests_per_minute"] if now - t < 60
        ]
        self.usage[model]["tokens_per_minute"] = [
            (t, tokens)
            for t, tokens in self.usage[model]["tokens_per_minute"]
            if now - t < 60
        ]

        # Clean per-day entries based on reset time
        reset_time = self._get_reset_time(model, "requests_per_day")
        if reset_time:
            # Remove entries older than last reset
            # reset_time is the NEXT reset, so last reset was 24 hours before that
            from datetime import timedelta

            last_reset = reset_time - timedelta(days=1)
            last_reset_timestamp = last_reset.timestamp()

            self.usage[model]["requests_per_day"] = [
                t
                for t in self.usage[model]["requests_per_day"]
                if t >= last_reset_timestamp
            ]
            self.usage[model]["tokens_per_day"] = [
                (t, tokens)
                for t, tokens in self.usage[model]["tokens_per_day"]
                if t >= last_reset_timestamp
            ]

    def _get_seconds_until_reset(self, model: str, quota_type: str) -> int:
        """Return seconds until the next quota reset for the given model/quota_type.

        Uses the timezone-aware datetime returned by `_get_reset_time` and the
        current time in that same timezone to compute the remaining seconds.
        If the reset time cannot be determined, returns a conservative default of 60s.
        """
        try:
            reset_dt = self._get_reset_time(model, quota_type)
            if not reset_dt:
                logging.debug(
                    "No reset time configured for model=%s quota=%s; defaulting to 60s",
                    model,
                    quota_type,
                )
                return 60

            # Compute now in the same timezone as the reset time
            now = datetime.now(reset_dt.tzinfo)
            seconds = (reset_dt - now).total_seconds()
            # Guard against negatives due to clock drift
            remaining = max(0, int(seconds))
            logging.debug(
                "Seconds until reset for model=%s quota=%s: %s (now=%s, reset=%s)",
                model,
                quota_type,
                remaining,
                now,
                reset_dt,
            )
            return remaining
        except Exception as e:
            logging.debug(
                "Failed to compute seconds until reset for model=%s quota=%s: %s; defaulting to 60s",
                model,
                quota_type,
                e,
            )
            return 60

    def check_and_wait(self, model: str, tokens: int = 0) -> bool:
        """Check quota limits and wait/abort as needed.

        Args:
            model: The model name to check quotas for
            tokens: Number of tokens for this request (optional)

        Returns:
            True if request can proceed, False if quota exceeded and should abort
        """
        if not self.config_loaded:
            # Warn once per model if config not loaded
            if model not in self.warned_models:
                logging.warning(
                    f"Quota tracking disabled for model '{model}': config not loaded"
                )
                self.warned_models.add(model)
            return True

        if model not in self.config.get("models", {}):
            if model not in self.warned_models:
                logging.warning(
                    f"No quota configuration found for model '{model}'. "
                    "Quota tracking disabled for this model."
                )
                self.warned_models.add(model)
            return True

        model_config = self.config["models"][model]
        quotas = model_config.get("quotas", {})
        quota_url = model_config.get("quota_url", "")

        # Clean old entries first
        self._clean_old_entries(model)

        # Check per-minute request limit
        if "requests_per_minute" in quotas:
            limit = quotas["requests_per_minute"]["limit"]
            current = len(self.usage[model]["requests_per_minute"])

            if current >= limit:
                logging.warning(
                    f"Model '{model}' per-minute request limit reached ({current}/{limit}). "
                    f"Waiting 60 seconds..."
                )
                time.sleep(60)
                self._clean_old_entries(model)

        # Check per-minute token limit
        if tokens > 0 and "tokens_per_minute" in quotas:
            limit = quotas["tokens_per_minute"]["limit"]
            current = sum(t for _, t in self.usage[model]["tokens_per_minute"])

            if current + tokens > limit:
                logging.warning(
                    f"Model '{model}' per-minute token limit would be exceeded "
                    f"({current + tokens}/{limit}). Waiting 60 seconds..."
                )
                time.sleep(60)
                self._clean_old_entries(model)

        # Check per-request token limit
        if tokens > 0 and "tokens_per_request" in quotas:
            limit = quotas["tokens_per_request"]["limit"]
            if tokens > limit:
                logging.critical(
                    f"Model '{model}' per-request token limit exceeded! "
                    f"Request: {tokens} tokens, Limit: {limit} tokens. "
                    f"Cannot proceed with this request."
                )
                return False

        # Check daily request limit
        if "requests_per_day" in quotas:
            limit = quotas["requests_per_day"]["limit"]
            current = len(self.usage[model]["requests_per_day"])

            if current >= limit:
                reset_time = self._get_reset_time(model, "requests_per_day")
                reset_str = (
                    reset_time.strftime("%Y-%m-%d %H:%M %Z")
                    if reset_time
                    else "unknown"
                )

                logging.critical(
                    f"Model '{model}' daily request limit exceeded!\n"
                    f"  Current: {current} requests\n"
                    f"  Limit: {limit} requests per day\n"
                    f"  Next reset: {reset_str}\n"
                    f"  Check quota at: {quota_url}"
                )
                return False

        # Check daily token limit
        if tokens > 0 and "tokens_per_day" in quotas:
            limit = quotas["tokens_per_day"]["limit"]
            current = sum(t for _, t in self.usage[model]["tokens_per_day"])

            if current + tokens > limit:
                reset_time = self._get_reset_time(model, "requests_per_day")
                reset_str = (
                    reset_time.strftime("%Y-%m-%d %H:%M %Z")
                    if reset_time
                    else "unknown"
                )

                logging.critical(
                    f"Model '{model}' daily token limit would be exceeded!\n"
                    f"  Current: {current} tokens\n"
                    f"  This request: {tokens} tokens\n"
                    f"  Limit: {limit} tokens per day\n"
                    f"  Next reset: {reset_str}\n"
                    f"  Check quota at: {quota_url}"
                )
                return False

        return True

    def record_request(self, model: str, tokens: int = 0):
        """Record a successful API request."""
        now = time.time()
        self.usage[model]["requests_per_minute"].append(now)
        self.usage[model]["requests_per_day"].append(now)

        if tokens > 0:
            self.usage[model]["tokens_per_minute"].append((now, tokens))
            self.usage[model]["tokens_per_day"].append((now, tokens))

        logging.debug(f"Recorded request for model '{model}' ({tokens} tokens)")


# Global quota tracker instance
_quota_tracker: Optional[QuotaTracker] = None


def get_quota_tracker() -> QuotaTracker:
    """Get or create the global quota tracker instance."""
    global _quota_tracker
    if _quota_tracker is None:
        _quota_tracker = QuotaTracker()
    return _quota_tracker


# ============================================================================
# END QUOTA TRACKING SYSTEM
# ============================================================================


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Please create a .env_gemini file with GEMINI_API_KEY=YOUR_API_KEY or set the environment variable."
        )
    return api_key


def image_mime_type(path: str) -> str:
    # Basic mapping based on file extension. Extend if needed.
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".jp2": "image/jp2",
    }.get(ext, "application/octet-stream")


def build_contents(prompt: str, image_paths: List[str]) -> List[Any]:
    """Return a list suitable for client.models.generate_content(contents=...)

    The first element is the prompt text, followed by Parts for each image.
    """
    contents: List[object] = [prompt]
    for path in image_paths:
        with open(path, "rb") as f:
            data = f.read()
        mime = image_mime_type(path)
        contents.append(types.Part.from_bytes(data=data, mime_type=mime))
    return contents


def validate_paths(paths: List[str]) -> List[str]:
    valid = []
    for p in paths:
        if not os.path.exists(p):
            logging.warning("File not found: %s", p)
            continue
        if not os.path.isfile(p):
            logging.warning("Not a file: %s", p)
            continue
        valid.append(p)
    return valid


def expand_path(path: str) -> List[str]:
    """If `path` is a directory, return a sorted list of image file paths inside it.

    If `path` is a file, return a single-item list containing that file.
    """
    if os.path.isdir(path):
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                ext = os.path.splitext(name)[1].lower()
                if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2"}:
                    entries.append(full)
        return entries
    return [path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A command-line tool for processing scanned images using Google's Gemini AI. Use 'extract' subcommand for OCR text extraction or 'proofread' to correct existing Markdown or hOCR drafts with image references.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",
        help="Set logging verbosity level (default: warning)",
    )
    parser.add_argument(
        "-l",
        "--log-to-file",
        nargs="?",
        const="",
        default=None,
        dest="log_dir",
        help="Log all messages to a file. If no path provided, logs to input directory. If path provided, logs to that directory.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract subcommand (existing behavior)
    p_extract = subparsers.add_parser("extract", help="Extract text from images (OCR)")
    p_extract.add_argument(
        "path",
        help="An image file or a directory containing image files to process",
    )
    p_extract.add_argument(
        "-o",
        "--output-file",
        dest="output_path",
        help=(
            "Path to write the Gemini response text. "
            "If omitted, a default file 'gemini-output-YYYY-MM-DDTHH:MM.txt' "
            "will be created in the directory provided in the 'path' argument."
        ),
    )
    p_extract.add_argument(
        "-s",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=30,
        help="Maximum number of images to send to Gemini in a single API call "
        "(default: 30, because Gemini seems to get stuck in loops around 35 images/pages)",
    )
    p_extract.add_argument(
        "-p",
        "--select-range",
        dest="select_range",
        action="store_true",
        help="Interactively select a range of files to process",
    )
    p_extract.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.0,
        help="Model temperature for generation (must be between 0.0 and 1.0, default: 0.0)",
    )
    p_extract.add_argument(
        "-m",
        "--max-attempts",
        dest="max_attempts",
        type=int,
        default=1,
        help="Maximum number of retry attempts for failures (must be 1 or greater, default: 1)",
    )

    # proofread subcommand
    p_proof = subparsers.add_parser(
        "proofread",
        help="Proofread and correct a Markdown or hOCR file using the provided images as reference",
    )
    p_proof.add_argument(
        "path",
        help="A directory containing subdirectories, each with one image file and one hOCR file to proofread",
    )
    p_proof.add_argument(
        "-o",
        "--output-dir",
        dest="output_path",
        help="Directory to write the corrected hOCR files. If omitted, uses the input path.",
    )
    p_proof.add_argument(
        "--debug",
        action="store_true",
        help="Print the <body> of the hOCR response to the terminal",
    )
    p_proof.add_argument(
        "-m",
        "--max-attempts",
        dest="max_attempts",
        type=int,
        default=1,
        help="Maximum number of retry attempts for validation failures (must be 1 or greater, default: 1)",
    )
    p_proof.add_argument(
        "-p",
        "--select-range",
        dest="select_range",
        action="store_true",
        help="Interactively select a range of subdirectories to process",
    )
    p_proof.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.0,
        help="Model temperature for generation (must be between 0.0 and 1.0, default: 0.0)",
    )

    # argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Validate temperature (must be between 0.0 and 1.0 inclusive)
    if hasattr(args, "temperature"):
        if not isinstance(args.temperature, (int, float)):
            parser.error("temperature must be a number")
        if not (0.0 <= args.temperature <= 1.0):
            parser.error("temperature must be between 0.0 and 1.0 (inclusive)")

    # Validate max_attempts (must be integer >= 1)
    if hasattr(args, "max_attempts"):
        if not isinstance(args.max_attempts, int):
            parser.error("max-attempts must be an integer")
        if args.max_attempts < 1:
            parser.error("max-attempts must be 1 or greater")

    return args


## Define some prompts here to switch between. prob better put in config file (or .env file?)
##   to keep text changes away from code change commits. keep at least a default prompt here
##   though for fallback purposes?
# TODO allow loading of text file, which contains only the prompt, no other text (i.e. all text in
#   file is passed as prompt)

prompt_1_3 = (
    "Act like a text scanner and transcribe the text in the image. "
    "Extract text as it is without analyzing it and without summarizing it. Treat "
    "all images as a whole document and analyze them accordingly. Think of it as a document with multiple "
    "pages, each image being a page. Understand page-to-page flow logically and semantically. "
    "Please mark the beginning and end of each page clearly in your response. "
    "Please print the page number according to the overall files processed "
    "(do not print the total file number), "
    "as well as the page number extracted from the current image. "
    "Please keep track of the chapters and sections in your response. "
    "Please put a header before the footnotes at the bottom of the page of `### FOOTNOTES` "
)

prompt_6 = (
    "Act like a text scanner and transcribe the text in the image."
    "Please make sure to transcribe the characters with their correct diacritics and spacing."
    "Treat images as part of one book and analyze them accordingly. "
    "Think of it as a document with multiple pages, each image being a page. "
    "Understand page-to-page flow logically and semantically."
    "Correct OCR-induced errors in the text, following these guidelines:\n"
    "1. Fix OCR-induced typos and errors:\n"
    "   - Fix common OCR errors (e.g., 'rn' misread as 'm')\n"
    "   - Use context and common sense to correct errors involving only one or two characters\n"
    "   - Only fix clear errors in one or two characters, don't alter the content of more than 3 characters in one place\n"
    "   - Do not add extra periods or any unnecessary punctuation\n"
    "2. Preserve original content:\n"
    "   - Keep all words and information from the original text\n"
    "   - Do not add any new information not present in the original text\n"
    "   - Maintain paragraph breaks\n"
    "   - Preserve all original formatting, including line breaks.\n"
    "To the outside of the main text lines there are sometimes numbers which mark subsections, Please keep"
    " those subsection numbers and surround the number with {} and leave them at the beginning or end of the line as they appear."
    "Do not include any introduction, explanation, or metadata. "
    "Please try to extract the page number from the image."
    "Please mark the beginning of each page clearly in your response"
    "with `### START FILE <BATCH NUMBER>, Extracted Page <PAGENUMBER>` where"
    " <BATCH NUMBER> is the sequential number of image in this batch and <PAGENUMBER> is the extracted page number if there is one. "
    "Please print the page number according to the overall files processed "
    "(do not print the total file number), "
    "as well as the page number extracted from the current image. "
    "Please keep track of the Part number (roman numerals) and the chapters headings in your response. "
    "Please mark off the beginning of a Part with `# PART <NUMBER>` "
    "and the beginning of a chapter with `## CHAPTER <NUMBER>` "
    "where <NUMBER> is the number of the chapter or part currently being processed."
    "Please put a header before the footnotes at the bottom of the page of `### FOOTNOTES <NUMBER>`"
    "where <NUMBER> is the number of the page currently being processed. "
    "Please make sure to recognize and transcribe all superscripts and subscripts and also footnotes."
)

DEFAULT_PROMPT = prompt_6


def get_client(api_key: str | None = None):
    """Create and return a Gemini client. Separated for testability."""
    if api_key is None:
        api_key = get_api_key()
    return genai.Client(api_key=api_key)


def prompt_user_for_range(items: List[str], item_type: str = "files") -> List[str]:
    """Prompt user to select a range of items to process.

    Args:
        items: List of items (file paths or directory names) to select from
        item_type: Description of items for display (e.g., "files", "subdirectories")

    Returns:
        Filtered list of items based on user selection

    Raises:
        SystemExit: If there is only one item or if the user input is invalid
    """
    if len(items) <= 1:
        logging.critical(
            f"Cannot select range: only {len(items)} {item_type} available. "
            "Range selection requires at least 2 items."
        )
        raise SystemExit(f"Cannot select range with only {len(items)} {item_type}.")

    # Extract numbers from basenames for display
    def extract_number(path):
        basename = os.path.basename(path)
        import re

        match = re.search(r"\d+", basename)
        return int(match.group()) if match else None

    first_item = os.path.basename(items[0])
    last_item = os.path.basename(items[-1])
    total = len(items)

    print(f"\n{'='*60}")
    print(f"Total {item_type} to process: {total}")
    print(f"First: {first_item}")
    print(f"Last:  {last_item}")
    print(f"{'='*60}")

    # Prompt the user for the starting index
    while True:
        start_input = input(f"Enter starting index (1-{total}, default=1): ").strip()
        if not start_input:
            start_idx = 0
            break
        if start_input.isdigit():
            start_idx = int(start_input) - 1  # Convert to 0-based index
            if 0 <= start_idx < total:
                break
        print(f"Invalid input. Please enter a number between 1 and {total}.")

    # Prompt for end index
    while True:
        end_input = input(
            f"Enter ending index ({start_idx + 1}-{total}, default={total}): "
        ).strip()
        if not end_input:
            end_idx = total - 1
            break
        if end_input.isdigit():
            end_idx = int(end_input) - 1  # Convert to 0-based index
            if start_idx <= end_idx < total:
                break
        print(
            f"Invalid input. Please enter a number between {start_idx + 1} and {total}."
        )

    selected = items[start_idx : end_idx + 1]
    selected_count = len(selected)

    print(
        f"\nSelected {selected_count} {item_type} from index {start_idx + 1} to {end_idx + 1}"
    )
    # TODO print image name too?
    print(f"First selected: {os.path.basename(selected[0])}")
    print(f"Last selected:  {os.path.basename(selected[-1])}")
    print(f"{'='*60}\n")

    while True:
        resp = input("Run with this range [(Y)es/(N)o/(Q)uit]? ").strip().lower()
        if resp in ("y", "yes"):
            # proceed with the currently selected range
            return selected
        if resp in ("n", "no"):
            # re-prompt the user to choose a new range
            print("Re-selecting range...")
            return prompt_user_for_range(items, item_type)
        if resp in ("q", "quit"):
            raise SystemExit("User quit.")
        print("Invalid input. Please enter Y, N, or Q.")

    # return selected


def determine_output_filepaths(
    input_path: str, output_dir: str | None = None, now: datetime | None = None
) -> tuple[str, str]:
    """Compute output file paths for text and prompt (default .txt for text)."""
    if output_dir is None:
        output_dir = (
            input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        )
    if now is None:
        now = datetime.now()
    dt = now.isoformat(timespec="minutes")
    return (
        os.path.join(output_dir, f"gemini-output-{dt}.txt"),
        os.path.join(output_dir, f"gemini-prompt-{dt}.txt"),
    )


def determine_proofread_output_filepaths(
    input_path: str,
    document_path: str,
    output_dir: str | None = None,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Compute output file paths for proofread command.

    - The output text file uses the SAME extension as the input document (.md/.markdown/.hocr).
    - The prompt file remains .txt.
    """
    if output_dir is None:
        output_dir = (
            input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        )
    if now is None:
        now = datetime.now()
    dt = now.isoformat(timespec="minutes")
    ext = os.path.splitext(document_path)[1].lower() or ".txt"
    # Normalize supported extensions; fallback to .txt if something unexpected slips through
    if ext not in {".md", ".markdown", ".hocr"}:
        ext = ".txt"
    return (
        os.path.join(output_dir, f"gemini-output-{dt}{ext}"),
        os.path.join(output_dir, f"gemini-prompt-{dt}.txt"),
    )


# Backward compatible alias used by tests to monkeypatch deterministic time
# Delegates to determine_output_filepaths.
# Keep signature and behavior identical.
def determine_output_paths(
    input_path: str, output_dir: str | None = None, now: datetime | None = None
) -> tuple[str, str]:
    return determine_output_filepaths(input_path, output_dir=output_dir, now=now)


def validate_response(
    response, batch_idx: int, num_batches: int, job_type=None, job_name=None
) -> tuple[str, str]:
    status = "ok"

    if not hasattr(response, "text"):
        # status = "Gemini response has no 'text' attribute"
        status = "no 'text' attribute"
        logging.error("Gemini response has no 'text' attribute: %r", response)
        raise SystemExit(2)

    response_text = response.text
    if response_text is None:
        status = "no text"
        if job_name:
            logging.error(
                "Gemini returned no text (None) for job %s.",
                job_name,
            )
        else:
            logging.error(
                "Gemini returned no text (None) for batch %d/%d.",
                batch_idx + 1,
                num_batches,
            )
        raise SystemExit(3)

    if isinstance(response_text, bytes):
        try:
            response_text = response_text.decode("utf-8", errors="replace")
        except Exception:
            logging.exception("Failed to decode bytes response as UTF-8.")
            raise SystemExit(4)
    elif not isinstance(response_text, str):
        status = "unexpected type"
        logging.error(
            "Gemini returned unexpected type for text: %s", type(response_text).__name__
        )
        raise SystemExit(5)

    if response_text.strip() == "":
        status = "empty string"
        if job_name:
            logging.warning(
                "Gemini returned an empty string for job %s.",
                job_name,
            )
        else:
            logging.warning(
                "Gemini returned an empty string for batch %d/%d.",
                batch_idx + 1,
                num_batches,
            )

    if job_type == "proofread":
        if not response_text.rstrip().endswith("</html>"):
            status = "no closing </html> tag"
            logging.info(f"Validation failed: does not end with </html>")
            # TODO logging.debug the last... 2 lines of the file?

    # TODO any other info useful for debug log? file size of output vs input? or do this after writing the file for ease?

    return response_text, status


def process_batches(
    client,
    image_paths: List[str],
    prompt: str,
    model: str,
    chunk_size: int,
    temperature: float = 0.0,
) -> str:
    """Send images to Gemini in batches and return the concatenated text."""
    total_images = len(image_paths)
    num_batches = (total_images + chunk_size - 1) // chunk_size
    logging.info(
        "Processing %d images in %d batch(es) of up to %d images each",
        total_images,
        num_batches,
        chunk_size,
    )

    quota_tracker = get_quota_tracker()

    all_text: list[str] = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]
        logging.info(
            "Processing batch %d/%d: images %d-%d",
            batch_idx + 1,
            num_batches,
            start_idx + 1,
            end_idx,
        )

        # Check quota before making request
        if not quota_tracker.check_and_wait(model):
            logging.critical(
                f"Quota exceeded for model '{model}'. Stopping batch processing."
            )
            raise SystemExit(1)

        contents = build_contents(prompt, batch_paths)

        # TODO test this actually works is correct config for api
        # Create config with temperature if not default
        config = None
        if temperature != 0.0:
            config = types.GenerateContentConfig(temperature=temperature)

        try:
            if config:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                response = client.models.generate_content(
                    model=model, contents=contents
                )
            # Record successful request
            quota_tracker.record_request(model)
        except Exception:
            logging.exception(
                "Gemini API call failed for batch %d/%d", batch_idx + 1, num_batches
            )
            raise SystemExit(1)

        batch_text, status_validation = validate_response(
            response, batch_idx, num_batches, job_type="OCR", job_name=batch_idx
        )
        all_text.append(batch_text)

    return "\n\n".join(all_text)


def proofread_page(
    client,
    image_path: str,
    prompt: str,
    model: str,
    config=None,
) -> tuple[str, str, int, str]:
    """Sends an hOCR file and an image to Gemini for proofreading.

    Returns a tuple (status, text):
    - status: 'ok' on success; 'error_429' for quota/backoff errors; 'error' for other failures; 'error_response' for invalid API response.
    - actual error response: empty string on success; error message on error
    - text: corrected hOCR HTML on success, otherwise an empty string.
    """

    response_error = ""

    logging.debug(
        "Processing %s image for proofreading",
        image_path,
    )
    # https://github.com/encode/httpx/discussions/2733
    logging.getLogger("httpx").setLevel("CRITICAL")
    logging.getLogger("httpx").propagate = False

    # Check quota before making request
    quota_tracker = get_quota_tracker()
    if not quota_tracker.check_and_wait(model):
        logging.critical(f"Quota exceeded for model '{model}'.")
        # TODO get wait time in seconds from quota tracker?
        return "error_quota", "Daily quota exceeded", 0, ""

    contents = build_contents(prompt, [image_path])

    # NOTE: do we want to try to check tokens first?
    # from: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count#gemini-get-token-count-samples-python_genai_sdk
    # they reccomend (for not just text-only prompts) to just run the query and get the results :/
    # the below they say is not reccomended?
    # response = client.models.count_tokens(
    #     model="gemini-2.5-flash",
    #     contents=contents,
    # )
    # print(response)

    try:
        if config:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
        else:
            response = client.models.generate_content(model=model, contents=contents)

        # Record successful request
        quota_tracker.record_request(model)
    except Exception as e:
        response_error = str(e)

        status_error = "error"  # default

        ## 429 error:
        # 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota,
        # please check your plan and billing details. For more information on this error,
        # head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage,
        # head to: https://ai.dev/usage?tab=rate-limit.\n* Quota exceeded for metric:
        # generativelanguage.googleapis.com/generate_content_free_tier_requests,
        # limit: 250\nPlease retry in 10.906087302s.', 'status': 'RESOURCE_EXHAUSTED',
        # 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric':
        # 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId':
        # 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global',
        # 'model': 'gemini-2.5-flash'}, 'quotaValue': '250'}]}, {'@type': 'type.googleapis.com/google.rpc.Help',
        # 'links': [{'description': 'Learn more about Gemini API quotas',
        # 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]},
        # {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '10s'}]}}

        # TODO: handle 503 error?
        # from log file: google.genai.errors.ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.', 'status': 'UNAVAILABLE'}}

        # Surface 429 (RESOURCE_EXHAUSTED) to caller so it can decide how/when to retry
        if hasattr(e, "code"):
            # Try to extract suggested retry delay (seconds) for logging purposes only
            retry_delay = 60
            # e.details[detail]["details"][0]["violations"][0]["quotaId"]

            try:
                if e.code == 503:
                    status_error = "error_503"
                    retry_delay = 120

                if e.code == 429:
                    status_error = "error_429"
                    if hasattr(e, "details") and e.details:
                        for detail in e.details:
                            if "details" in e.details[detail].keys():
                                for quota_detail in e.details[detail]["details"]:
                                    if (
                                        quota_detail.get("@type")
                                        == "type.googleapis.com/google.rpc.QuotaFailure"
                                    ):
                                        if "violations" in quota_detail.keys():
                                            for quota_violation in quota_detail[
                                                "violations"
                                            ]:
                                                if "quotaId" in quota_violation.keys():
                                                    if (
                                                        quota_violation["quotaId"]
                                                        == "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                                                    ):
                                                        retry_delay = quota_tracker._get_seconds_until_reset(
                                                            "gemini-2.5-flash",
                                                            "requests_per_day",
                                                        )
                                                        # retry_delay = 52

                                                        seconds_in_23_hours = (
                                                            23 * 60 * 60
                                                        )
                                                        if (
                                                            retry_delay
                                                            > seconds_in_23_hours
                                                        ):
                                                            # ok wait another full day? I don't buy that. wait an hour instead
                                                            retry_delay = 1 * 60 * 60

                                                        logging.critical(
                                                            "Over requests per day quota, waiting for {}s, until {}".format(
                                                                retry_delay,
                                                                quota_tracker._get_reset_time(
                                                                    "gemini-2.5-flash",
                                                                    "requests_per_day",
                                                                ),
                                                            )
                                                        )
                                                        # time.sleep(retry_delay)
                                                        # TODO remove return?
                                                        # return (
                                                        #     "error_429",
                                                        #     response_error,
                                                        #     retry_delay,
                                                        #     "",
                                                        # )
                                                    elif (
                                                        quota_violation["quotaId"]
                                                        == "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                                                    ):
                                                        # TODO: is this above, the right error for this minute?
                                                        retry_delay = 60
                                                        # TODO This below might be userful here? or just wait 60s??
                                                        # if (
                                                        #     isinstance(
                                                        #         e.details[detail], dict #TODO need to fix this then
                                                        #     )
                                                        #     and detail.get("@type")
                                                        #     == "type.googleapis.com/google.rpc.RetryInfo"
                                                        # ):
                                                        #     delay_str = detail.get(
                                                        #         "retryDelay", "60s"
                                                        #     )
                                                        #     import re
                                                        #
                                                        #     match = re.match(
                                                        #         r"(\d+(?:\.\d+)?)s",
                                                        #         delay_str,
                                                        #     )
                                                        #     if match:
                                                        #         retry_delay = int(
                                                        #             float(match.group(1))
                                                        #             + 1
                                                        #         )

                                                        # return (
                                                        #     "error_429",
                                                        #     response_error,
                                                        #     retry_delay, #default is 60
                                                        #     "",
                                                        # )
            except Exception:
                pass

            image_filename = os.path.basename(image_path)
            image_subdirname = os.path.basename(
                os.path.normpath(os.path.dirname(image_path))
            )
            logging.warning(
                "Gemini API %s ERROR for %s in %s — waiting ~%ss before retrying",
                str(e.code),
                image_filename,
                image_subdirname,
                retry_delay,
            )
            # should only get here if 429 error is of day or minute limit, just wait a minute??
            return status_error, response_error, retry_delay, ""

        # Other exceptions: let caller treat as retryable error
        logging.exception(
            "Gemini API call failed for proofreading %s",
            image_path,
        )
        return "error", response_error, 0, ""

    # Validate and convert response to text
    try:
        batch_text, status_validation = validate_response(
            response,
            batch_idx=1,
            num_batches=1,
            job_type="proofread",
            job_name=image_path,
        )
    except SystemExit:
        # Convert validation failure into a status for the caller
        return "error", response_error, 0, ""

    return status_validation, response_error, 0, batch_text


def build_proofread_prompt(draft_text: str, fmt: str) -> str:
    """Build a proofreading prompt for the given document format.

    `fmt`: "markdown" or "hocr"
    """
    fmt = fmt.lower()
    if fmt == "markdown":
        return (
            "You are an expert proofreader for OCR outputs. Given a draft in Markdown and reference images "
            "that contain the true source pages, correct mistakes in the draft strictly based on the images.\n\n"
            "Rules:\n"
            "- Preserve Markdown structure (headings, lists, emphasis).\n"
            "- Do not invent content; only correct what is clearly wrong per the images.\n"
            "- Keep paragraph breaks and footnotes, preserving superscripts/subscripts where applicable.\n"
            "- Maintain any explicit page or section markers if present.\n"
            "- Respond ONLY with the corrected Markdown, no explanations.\n\n"
            "Draft to correct (Markdown):\n\n" + draft_text
        )
    elif fmt == "hocr":
        # TODO try to keep weird margin line numbers?? it seems to be doing ok with them...
        return (
            "You are an expert proofreader for OCR outputs. Given the `hOCR HTML document to correct` that is copied"
            " below, please ignore all xml content and only correct the 'text' objects from the `hOCR HTML document to "
            "correct`. To correct the 'text' objects from the `hOCR HTML document to correct` that is copied below, "
            "please use the provided true source image to strictly make changes only to the 'text' content of the "
            "`hOCR HTML document to correct` based ONLY on the provided true source image."
            "Please make sure to check that there are the correct diacritic marks on all characters,"
            "as could be expected for Arabic transliterated words."
            "Please also pay attention to superscripts after words which mark footnotes, and please pay attention"
            "to text at the bottom of the page which are footenotes corresponding to the superscript numbers on that"
            "page. \n\n"
            "Please transcribe back the full hOCR HTML document to correct that is copied below, with corrected 'text'"
            "objects. Do not include any explanations. Respond ONLY with the corrected hOCR HTML.\n\n"
            "`hOCR HTML document to correct`:\n\n" + draft_text
        )
    else:
        raise ValueError(f"Unsupported document format for proofreading: {fmt}")


def detect_document_format(document_path: str) -> str:
    """Detect the document format from its file extension.

    Returns: "markdown" or "hocr". Raises SystemExit for unsupported formats.
    """
    ext = os.path.splitext(document_path)[1].lower()
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext == ".hocr":
        return "hocr"
    raise SystemExit(
        f"Unsupported document type: '{ext}'. Supported: .md, .markdown, .hocr"
    )


def run_proofread(
    path: str,
    output_path: str | None = None,
    model: str = "gemini-2.5-flash",
    client=None,
    debug: bool = False,
    max_attempts: int = 2,
    select_range: bool = False,
    temperature: float = 0.0,
) -> tuple[str, str, str, str]:
    # TODO does this need the default value also?? for max_attempts
    """Proofread hOCR documents in subdirectories using the provided images as reference.

    Expects the path to be a directory of subdirectories, each containing one image and one hOCR file.

    Returns ("", "", output_path, "").
    """
    if not os.path.isdir(path):
        raise SystemExit(f"Path {path} is not a directory.")

    if output_path is None:
        output_path = path

    os.makedirs(output_path, exist_ok=True)

    subdirs = sorted(
        [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    )

    # Prompt user for range selection if requested
    if select_range:
        subdirs = prompt_user_for_range(subdirs, item_type="subdirectories")

    client = client or get_client()

    pbar = tqdm(subdirs, desc="Proofreading pages", dynamic_ncols=True)
    for subdir in pbar:
        subdir_path = os.path.join(path, subdir)

        # Find image files
        image_files = [
            f
            for f in os.listdir(subdir_path)
            if os.path.splitext(f)[1].lower()
            in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2"}
        ]
        if len(image_files) != 1:
            logging.warning(
                f"Expected 1 image in {subdir_path}, found {len(image_files)}"
            )
            continue
        image_path = os.path.join(subdir_path, image_files[0])
        image_filename = image_files[0]

        # Update tqdm bar with the current image filename
        # pbar.set_description(f"Proofing: {image_filename}")
        # Update tqdm bar with the current subdir name
        pbar.set_description(f"Proofing: {subdir}")

        # Find hOCR files
        hocr_files = [f for f in os.listdir(subdir_path) if f.endswith(".hocr")]
        if len(hocr_files) != 1:
            logging.info(
                f"Expected 1 hocr in {subdir_path}, found {len(hocr_files)}, skipping..."
            )
            continue
        hocr_filepath = os.path.join(subdir_path, hocr_files[0])
        hocr_filename = hocr_files[0]
        out_filename = hocr_filename.replace(".hocr", "-proofread.hocr")
        out_filepath = os.path.join(
            subdir_path, out_filename
        )  # save to same dir as hocr
        # out_filepath = os.path.join(path, out_filename) #save to basedir, harder to compare later?

        with open(hocr_filepath, "r", encoding="utf-8") as f:
            draft = f.read()
        prompt = build_proofread_prompt(draft_text=draft, fmt="hocr")

        # Live timer thread for tqdm

        # Try proofreading with validation and retry logic
        corrected = None
        elapsed = 0  # Initialize elapsed before loop
        # for attempt in range(max_attempts):
        attempt = 0
        temperatute_increase_per_attempt = 0.1
        while attempt < max_attempts:
            # Create a string to use for filenames with datetime (no seconds, no colons)
            attempt_time_start = datetime.now()
            attempt_time_start_string = attempt_time_start.strftime(
                "%Y-%m-%dT%H%M"
            )  # ISO format without seconds and without colons

            out_temperature_filename = hocr_filename.replace(
                ".hocr",
                "-proofread.lasttemp",
            )
            out_temperature_filepath = os.path.join(
                subdir_path, out_temperature_filename
            )  # save to same dir as hocr

            # If a temperature override file exists, try to load and validate it.
            if os.path.exists(out_temperature_filepath) and os.path.isfile(
                out_temperature_filepath
            ):
                try:
                    with open(out_temperature_filepath, "r", encoding="utf-8") as tf:
                        txt = tf.read().strip()
                    if txt != "":
                        val = float(txt)
                        if 0.0 <= val <= 1.0:
                            logging.info(
                                "Loaded temperature %s from `%s`",
                                val,
                                out_temperature_filepath,
                            )
                            # temperature = val + temperatute_increase_per_attempt #load the last temperature and then bump it up by one attempt
                            # VS...
                            # load the last temperature, but don't increase it if it is already > 0, just try again
                            if val == 0.0:
                                temperature = temperatute_increase_per_attempt
                            else:
                                temperature = val
                        else:
                            logging.warning(
                                "Ignored temperature in `%s`: value %s not in [0.0, 1.0]",
                                out_temperature_filepath,
                                val,
                            )
                    else:
                        logging.debug(
                            "Temperature file `%s` is empty, ignoring",
                            out_temperature_filepath,
                        )
                except Exception:
                    logging.warning(
                        "Failed to read/parse temperature file `%s`, ignoring",
                        out_temperature_filepath,
                        exc_info=True,
                    )

            # Increase temperature by 0.1 after the first attempt, capped at 1.0
            per_attempt_temp = temperature + temperatute_increase_per_attempt * attempt
            if per_attempt_temp > 1.0:
                per_attempt_temp = 1.0

            config = types.GenerateContentConfig(
                system_instruction=(
                    "you understand the hOCR format and can easily extract only the scanned text "
                    "from the hocr document (the `ocrx_word` content)"
                ),
                temperature=per_attempt_temp,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                should_return_http_response=False,
            )
            logging.debug(
                f"{image_filename}: Using temperature={per_attempt_temp:.2f} on attempt {attempt + 1}/{max_attempts}"
            )

            timer_running = True
            start_time = time.time()

            def update_timer():
                while timer_running:
                    elapsed_local = time.time() - start_time
                    pbar.set_postfix(
                        {
                            "job_time": f"{elapsed_local:.2f}s",
                            "attempt": f"{attempt+1}/{max_attempts}",
                        }
                    )
                    time.sleep(0.05)

            timer_thread = threading.Thread(target=update_timer)
            timer_thread.start()
            status_validation, status_error, retry_additional_delay, corrected = (
                proofread_page(client, image_path, prompt, model, config=config)
            )

            timer_running = False
            timer_thread.join()
            elapsed = time.time() - start_time
            pbar.set_postfix(
                {
                    "job_time": f"{elapsed:.2f}s",
                    "attempt": f"{attempt+1}/{max_attempts}",
                }
            )

            # write out file, or figure out if this request should could against attempts
            codes_retry = ["error_429", "error_503"]
            if status_validation == "ok":
                # Save corrected hOCR
                write_text(out_filepath, corrected)

                if elapsed < 60:
                    pbar.set_description(
                        f"Waiting for {60 - elapsed}s before processing next page"
                    )
                    time.sleep(60 - elapsed)

                break
            elif status_validation not in codes_retry:
                # for any error except over quota/rate limit, count it as an attempt
                attempt += 1

                # create filename for storing temperature value
                out_temperature_filename = hocr_filename.replace(
                    ".hocr",
                    "-proofread.lasttemp",
                )
                out_temperature_filepath = os.path.join(
                    subdir_path, out_temperature_filename
                )  # save to same dir as hocr
                # Save temperature value that failed validation
                write_text(out_temperature_filepath, str(per_attempt_temp))

                # only write errors out if NOT an over quota error.
                # TODO make this command line arg? or at least more visible somewhere in script?
                write_fails = True
                if write_fails:
                    # create filename for failed hocr that does not stop this script from reprocessing this directory
                    out_failed_filename = hocr_filename.replace(
                        ".hocr",
                        "-proofread.hocr.{0}.{1}.failed".format(
                            attempt_time_start_string, attempt
                        ),
                    )
                    out_failed_filepath = os.path.join(
                        subdir_path, out_failed_filename
                    )  # save to same dir as hocr

                    # Save hOCR response that failed validation
                    write_text(out_failed_filepath, corrected)

                write_errors = True
                if write_errors:
                    out_error_filename = hocr_filename.replace(
                        ".hocr",
                        "-proofread.err.{0}.{1}.failed".format(
                            attempt_time_start_string, attempt
                        ),
                    )
                    out_error_filepath = os.path.join(
                        subdir_path, out_error_filename
                    )  # save to same dir as hocr

                    # Save error for hOCR response that failed validation
                    write_text(out_error_filepath, status_error)

            # Retry logic (existing behavior)
            # if attempt < max_attempts - 1 or status_validation == "error_429": #this was for the for loop
            if attempt < max_attempts or status_validation in codes_retry:

                retry_delay = 60 * attempt + retry_additional_delay

                # we don't print an error message, so update the progress bar to make it more obvious
                # (than just the attempt number at the end of the bar)
                if retry_delay == 0:
                    pbar.set_description(f"Retrying: {image_filename}")
                    break

                timer_running = True
                start_time = time.time()
                timer_thread_waiting = threading.Thread(target=update_timer)
                timer_thread_waiting.start()

                if status_validation == "error_429":
                    pbar.set_description(f"OVERQUOTA! Waiting {retry_delay}s")
                else:
                    pbar.set_description(
                        f"Retrying: {image_filename} after {retry_delay}s"
                    )
                time.sleep(retry_delay)
                timer_running = False
                timer_thread_waiting.join()

                # pbar description is set outside the while loop, so reset it here
                pbar.set_description(f"Retrying: {image_filename}")
            else:
                logging.error(
                    f"Failed validation after {max_attempts} attempts for {hocr_filename}"
                )

        # TODO: run some quick check on file.
        # - check it is within x% of original (15%? 5%?)
        # - check it is not empty ✓ DONE
        # - check it ends with </html> (or same line as prev file... ever not </html>?) ✓ DONE

        # NOTE: most of this should prob be done in postproofread.py, but some could be done here on single page basis?
        # TODO post process text? optionally delete empty spans?
        # 1st delete empty <span class='ocrx_word' id='word_3_6' title='bbox 3 109 67 189;x_wconf 0;x_font Century_Schoolbook_L_Bold' style='font-family:Century_Schoolbook_L_Bold'></span>
        # 2nd delete empty <span class='ocr_line' title="bbox 3 109 67 189; baseline -0.006711 0; x_x_height 81; x_asc_height 77">NEWLINE</span>
        # NOTE: don't delete empty pages! keep them for correct sequencing
        # TODO MUST remove the comment line. or at least move it somewhere else... it seems that putting it after the div page is fine...
        # TODO MAYBE? add the font crap back in, see if that loads in scribe ocr well?

        if debug:
            # Extract and print <body>
            try:
                root = ET.fromstring(corrected)
                body = root.find(".//{http://www.w3.org/1999/xhtml}body") or root.find(
                    ".//body"
                )
                if body is not None:
                    print(ET.tostring(body, encoding="unicode"))
                else:
                    print(f"No <body> found in {subdir}")
            except ET.ParseError:
                print(f"Failed to parse hOCR for {subdir}")

    return "", "", output_path, ""


def write_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        logging.exception("Failed to write output to %s", path)
        raise SystemExit(6)


def run_ocr(
    path: str,
    output_path: str | None = None,
    chunk_size: int = 30,
    prompt: str | None = None,
    model: str = "gemini-2.5-flash",
    client=None,
    select_range: bool = False,
    temperature: float = 0.0,
) -> tuple[str, str, str, str]:
    """High-level API used by CLI and tests. Returns (text, prompt_used, text_path, prompt_path)."""
    # Expand and validate input
    expanded = expand_path(path)
    image_paths = validate_paths(expanded)
    if not image_paths:
        raise SystemExit("No valid image files provided.")

    # Prompt user for range selection if requested
    if select_range:
        image_paths = prompt_user_for_range(image_paths, item_type="files")

    # Client and prompt
    client = client or get_client()
    prompt_to_use = prompt or DEFAULT_PROMPT

    # Process
    text = process_batches(
        client,
        image_paths,
        prompt_to_use,
        model=model,
        chunk_size=chunk_size,
        temperature=temperature,
    )

    # Output files
    out_text_filepath, out_prompt_filepath = determine_output_filepaths(
        path, output_dir=output_path
    )
    write_text(out_text_filepath, text)
    write_text(out_prompt_filepath, prompt_to_use)

    return text, prompt_to_use, out_text_filepath, out_prompt_filepath


def setup_file_logging(
    log_dir: str | None, input_path: str, command: str
) -> str | None:
    """Set up file logging if requested.

    Args:
        log_dir: Directory to save logfile, or empty string for input dir, or None to disable
        input_path: The input path from the command line (used to determine default log location)
        command: The subcommand being run (extract or proofread)

    Returns:
        Path to the log file, or None if file logging is disabled
    """
    if log_dir is None:
        return None

    # Determine log directory
    if log_dir == "":
        # Use input directory
        if os.path.isdir(input_path):
            log_dir = input_path
        else:
            log_dir = os.path.dirname(input_path) or "."

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with datetime (no seconds, no colons)
    now = datetime.now()
    dt = now.strftime("%Y-%m-%dT%H%M")  # ISO format without seconds and without colons
    log_filename = f"gemini_{command}_{dt}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Add file handler to root logger
    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    return log_filepath


def main():
    args = parse_args()

    # Configure logging: root logger captures everything; console shows per --verbosity
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs in tests or re-entrancy
    while root_logger.handlers:
        root_logger.handlers.pop()
    root_logger.setLevel(logging.DEBUG)  # capture all messages globally

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_map[args.verbosity])
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console_handler)

    # Set up file logging if requested (file gets EVERYTHING at DEBUG level)
    log_filepath = setup_file_logging(
        log_dir=args.log_dir, input_path=args.path, command=args.command or "extract"
    )
    if log_filepath:
        logging.info(f"Logging to file: {log_filepath}")

    if args.command == "proofread":
        text, prompt_used, out_text, out_prompt = run_proofread(
            path=args.path,
            output_path=args.output_path,
            debug=getattr(args, "debug", False),
            max_attempts=getattr(args, "max_attempts", 2),
            select_range=getattr(args, "select_range", False),
            temperature=getattr(args, "temperature", 0.0),
        )
    else:  # extract (default)
        text, prompt_used, out_text, out_prompt = run_ocr(
            path=args.path,
            output_path=args.output_path,
            chunk_size=getattr(args, "chunk_size", 30),
            prompt=prompt_6,
            select_range=getattr(args, "select_range", False),
            temperature=getattr(args, "temperature", 0.0),
        )

    # Print the response text (the extracted document text)
    print(text)
    logging.info("Saved Gemini response to %s", out_text)
    logging.info("Saved Gemini prompt to %s", out_prompt)


if __name__ == "__main__":
    main()
