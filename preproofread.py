#!/usr/bin/env python3
"""
preproofread.py

Prepare batched directories of images and sliced hOCR files for proofreading.

- Accepts: input hOCR file and the directory with the source images (e.g., .png, .jpg, .jpeg, .tif, .tiff, .bmp, .gif, .webp, .jp2).
- Validates that the number of images equals the number of <div class='ocr_page' ...> pages in the hOCR file.
- Splits into batches (default size 20) and creates subdirectories in the specified output directory.
- Each batch directory contains:
  - Copies of the corresponding images for that batch.
  - A new hOCR file preserving the original <head> content and <body ...> attributes but including only the relevant page divs.
  - Inside each page div, a single comment line is inserted after the opening tag, stating the overall page number and the source image filename.

This script never modifies the original hOCR or images; it only creates copies in the output directory.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

# Supported image extensions (lowercase, without the dot)
IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp", "jp2"}


@dataclass
class HOCRStructure:
    prefix_before_body: str  # Everything up to and including <head>...</head>, and any tags before <body>
    body_open_tag: str  # The complete <body ...> opening tag
    pages_html: List[
        str
    ]  # Each entry is a full <div class="ocr_page" ...> ... </div> block


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_images_sorted(images_dir: Path) -> List[Path]:
    files = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in IMAGE_EXTS
    ]
    files.sort(key=lambda p: p.name)
    return files


def extract_hocr_structure(hocr_html: str) -> HOCRStructure:
    # Capture everything before the first <body ...>, including <head> content
    body_open_match = re.search(
        r"<body\b[^>]*>", hocr_html, flags=re.IGNORECASE | re.DOTALL
    )
    if not body_open_match:
        raise ValueError("Input hOCR does not contain a <body> tag.")

    body_open_tag = body_open_match.group(0)
    prefix_before_body = hocr_html[: body_open_match.start()].rstrip()
    prefix_before_body = remove_meta_tags(prefix_before_body)

    # Extract body content (between <body ...> and </body>)
    body_close_match = re.search(r"</body\s*>", hocr_html, flags=re.IGNORECASE)
    if not body_close_match:
        raise ValueError("Input hOCR does not contain a closing </body> tag.")

    body_inner = hocr_html[body_open_match.end() : body_close_match.start()]

    # Find all <div class="ocr_page" ...> ... </div> blocks in body_inner
    pages_html = extract_ocr_page_divs(body_inner)
    if not pages_html:
        raise ValueError(
            "No <div class='ocr_page' ...> elements found in the hOCR file."
        )

    return HOCRStructure(
        prefix_before_body=prefix_before_body,
        body_open_tag=body_open_tag,
        pages_html=pages_html,
    )


def extract_ocr_page_divs(body_inner: str) -> List[str]:
    """Return a list of full HTML snippets for each ocr_page <div> block.

    We find each opening <div> that has a class attribute containing the word ocr_page
    (quotes can be single or double, class can contain multiple tokens), and then
    collect text until its matching closing </div> by tracking nested <div>.
    """
    # Pattern to find candidate opening ocr_page divs
    open_re = re.compile(
        r"<div\b[^>]*class\s*=\s*([\"'])[^\"']*\bocr_page\b[^>]*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Generic patterns to track nested divs
    any_open_div = re.compile(r"<div\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
    any_close_div = re.compile(r"</div\s*>", flags=re.IGNORECASE)

    pages: List[str] = []
    pos = 0
    while True:
        m = open_re.search(body_inner, pos)
        if not m:
            break
        start = m.start()
        # Track nested divs to find the matching closing </div>
        depth = 0
        i = start
        while True:
            next_open = any_open_div.search(body_inner, i)
            next_close = any_close_div.search(body_inner, i)
            if next_close is None and next_open is None:
                raise ValueError(
                    "Malformed hOCR: unmatched <div> for an ocr_page block."
                )
            # Choose the earliest tag
            if next_close is None or (
                next_open is not None and next_open.start() < next_close.start()
            ):
                depth += 1
                i = next_open.end()
            else:
                depth -= 1
                i = next_close.end()
                if depth == 0:
                    # Completed this ocr_page div
                    end = i
                    snippet = body_inner[start:end]
                    pages.append(snippet)
                    pos = end  # continue search after this block
                    break
    return pages


def assemble_hocr(
    prefix_before_body: str,
    body_open_tag: str,
    page_blocks_with_comments: Sequence[str],
) -> str:
    parts = []
    # Ensure the prefix ends with a newline
    prefix = prefix_before_body.rstrip() + "\n"
    parts.append(prefix)
    parts.append(body_open_tag)
    parts.append("\n")
    # Insert page blocks with comments already prepared
    for block in page_blocks_with_comments:
        parts.append(block)
        if not block.endswith("\n"):
            parts.append("\n")
    parts.append("</body>\n</html>\n")
    return "".join(parts)


def split_batches(total: int, batch_size: int) -> List[Tuple[int, int]]:
    """Return a list of (start, end) index pairs for batching [0, total).
    Each pair is inclusive start, exclusive end.
    """
    batches: List[Tuple[int, int]] = []
    i = 0
    while i < total:
        j = min(i + batch_size, total)
        batches.append((i, j))
        i = j
    return batches


def create_batches(
    hocr: HOCRStructure,
    images: List[Path],
    output_dir: Path,
    batch_size: int,
    hocr_basename: Optional[str] = None,
) -> None:
    total_pages = len(hocr.pages_html)
    assert total_pages == len(images)

    batches = split_batches(total_pages, batch_size)

    # Determine the basename for hOCR outputs (input file name). If not provided, use a generic base.
    if hocr_basename is None:
        hocr_basename = "batch.hocr"

    def derive_batch_hocr_filename(
        input_name: str, start_inclusive_zero: int, end_exclusive_zero: int
    ) -> str:
        """Build output hOCR filename from the input filename and the 1-based page range.
        - Strips a trailing "-all" or "_all" right before the .hocr extension (case-insensitive).
        - Appends _<start>-<end> before the .hocr extension, where start/end are 1-based and inclusive.
        """
        name = input_name
        # Ensure .hocr extension handling is case-insensitive
        lower = name.lower()
        if lower.endswith(".hocr"):
            stem = name[:-5]  # remove .hocr
        else:
            # If not .hocr, treat the entire name as stem and append .hocr
            stem = name

        lower_stem = stem.lower()
        if lower_stem.endswith("-all"):
            stem = stem[:-4]
        elif lower_stem.endswith("_all"):
            stem = stem[:-4]
        start1 = start_inclusive_zero + 1
        end1 = end_exclusive_zero  # since 'end' is exclusive, this is inclusive 'end' in 1-based
        if start1 == end1:
            return f"{stem}_{start1}.hocr"
        else:
            return f"{stem}_{start1}-{end1}.hocr"

    for batch_idx, (start, end) in enumerate(
        tqdm(batches, desc="Creating batches"), start=1
    ):
        # batch_dir = output_dir / f"batch-{batch_idx:03d}"
        if start + 1 == end:
            batch_dir = output_dir / f"batch_{start+1:03d}"
        else:
            batch_dir = output_dir / f"batch_{start+1:03d}-{end:03d}"

        batch_dir.mkdir(parents=True, exist_ok=True)

        # Copy images for this batch
        for img in images[start:end]:
            shutil.copy2(img, batch_dir / img.name)

        # Prepare hOCR content for this batch with comments
        page_blocks_with_comments: List[str] = []
        for page_index in range(start, end):
            # Overall (1-based) page number
            overall_page_num = page_index + 1
            image_name = images[page_index].name
            comment = (
                f"<!-- overall page {overall_page_num}: source image {image_name} -->"
            )
            page_html = hocr.pages_html[page_index]
            # Insert comment inside the div, after the opening tag
            match = re.search(r"<div\b[^>]*>", page_html, re.IGNORECASE)
            if match:
                insert_pos = match.end()
                modified_page_html = (
                    page_html[:insert_pos] + "\n" + comment + page_html[insert_pos:]
                )
            else:
                modified_page_html = page_html  # fallback
            page_blocks_with_comments.append(modified_page_html)

        batch_hocr = assemble_hocr(
            prefix_before_body=hocr.prefix_before_body,
            body_open_tag=hocr.body_open_tag,
            page_blocks_with_comments=page_blocks_with_comments,
        )

        batch_hocr_name = derive_batch_hocr_filename(hocr_basename, start, end)
        out_hocr_path = batch_dir / batch_hocr_name
        write_text(out_hocr_path, batch_hocr)


def remove_meta_tags(html: str) -> str:
    """Remove specific meta tags from the HTML head section."""
    lines = html.split("\n")
    filtered = []
    for line in lines:
        if re.search(
            r"<meta\b[^>]*name\s*=\s*['\"](font-metrics|layout|layout-data-table)['\"]",
            line,
            re.IGNORECASE,
        ):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an hOCR file and source images into proofreading batches."
    )
    parser.add_argument(
        "--hocr", "-f", type=Path, required=True, help="Path to the input .hocr file"
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=Path,
        required=True,
        help="Path to the directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to create batch subdirectories in",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=20,
        help="Number of pages/images per batch (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    hocr_path: Path = args.hocr
    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir
    batch_size: int = args.batch_size

    if batch_size <= 0:
        print("Error: --batch-size must be a positive integer.", file=sys.stderr)
        return 2

    if not hocr_path.is_file():
        print(f"Error: hOCR file not found: {hocr_path}", file=sys.stderr)
        return 2
    if not images_dir.is_dir():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        hocr_html = read_text(hocr_path)
        hocr_struct = extract_hocr_structure(hocr_html)
    except Exception as e:
        print(f"Error: failed to parse hOCR file: {e}", file=sys.stderr)
        return 2

    images = list_images_sorted(images_dir)
    print(f"Found {len(images)} images in {images_dir}")
    if not images:
        print(
            f"Error: no supported images found in directory: {images_dir}",
            file=sys.stderr,
        )
        return 2

    pages_count = len(hocr_struct.pages_html)
    if len(images) != pages_count:
        print(
            "Error: mismatch between number of images and number of pages in hOCR. "
            f"images: {len(images)}, hOCR pages: {pages_count}.\n"
            "Assumption: pages in hOCR are ordered to match the filename-sorted images.",
            file=sys.stderr,
        )
        return 2

    # Name the hOCR file inside each batch after the original file, if present
    hocr_basename = hocr_path.name if hocr_path.name else "batch.hocr"

    create_batches(
        hocr_struct, images, output_dir, batch_size, hocr_basename=hocr_basename
    )

    print(
        f"Created {len(split_batches(pages_count, batch_size))} batch(es) in: {output_dir}\n"
        f"Images: {len(images)} | Pages: {pages_count} | Batch size: {batch_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
