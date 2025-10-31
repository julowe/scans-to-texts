#!/usr/bin/env python3
"""
postproofread.py - Merge proofread HOCR files and generate diff reports

This script processes the output of the proofreading workflow by:
1. Comparing original and proofread HOCR files to generate diffs
2. Merging proofread HOCR files back into a single multi-page HOCR file
3. Generating diff reports (text and HTML) showing all changes

The script expects an input directory containing batch subdirectories, where each
batch contains:
- Source images (for batch size determination)
- Original HOCR file (single page or batch)
- Proofread HOCR file (with corrections)
"""

# TODO: highlight in darker color for added/removed text

import argparse
import difflib
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
from bs4 import BeautifulSoup

# Optional progress bar for long diffs; fall back gracefully if tqdm isn't installed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def detect_batch_size(batch_dir: Path) -> int:
    """
    Determine the batch size from the directory name or the number of source images.

    Args:
        batch_dir: Path to a batch directory

    Returns:
        Number of pages in the batch
    """
    # Try to extract batch size from directory name (e.g., "batch_0001_5" means 5 pages)
    dir_name = batch_dir.name
    match = re.search(r'batch_\d+_(\d+)', dir_name)
    if match:
        return int(match.group(1))

    # Count source images (common image extensions)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jp2'}
    images = [f for f in batch_dir.iterdir()
              if f.is_file() and f.suffix.lower() in image_extensions]

    if images:
        return len(images)

    # Default to 1 if we can't determine
    return 1


def find_hocr_files(batch_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find the original and proofread HOCR files in a batch directory.

    Args:
        batch_dir: Path to a batch directory

    Returns:
        Tuple of (original_hocr_path, proofread_hocr_path)
    """
    hocr_files = list(batch_dir.glob('*.hocr'))

    if not hocr_files:
        return None, None

    if len(hocr_files) == 1:
        # Assume it's the proofread version
        return None, hocr_files[0]

    # Try to identify which is original and which is proofread
    original = None
    proofread = None

    for hocr in hocr_files:
        name_lower = hocr.name.lower()
        if 'proofread' in name_lower or 'corrected' in name_lower or 'edited' in name_lower:
            proofread = hocr
        elif 'original' in name_lower:
            original = hocr
        else:
            # FIXME ?
            # If we can't tell, use file modification time
            if original is None:
                original = hocr
            else:
                # The newer one is likely the proofread version
                if hocr.stat().st_mtime > original.stat().st_mtime:
                    proofread = hocr
                else:
                    proofread = original
                    original = hocr

    # If we only found one file, treat it as proofread
    if proofread is None and original is not None:
        proofread = original
        original = None

    return original, proofread


def generate_diff(original_path: Path, proofread_path: Path, diff_output_path: Path) -> List[str]:
    """
    Generate a unified diff between original and proofread HOCR files.

    Args:
        original_path: Path to original HOCR file
        proofread_path: Path to proofread HOCR file
        diff_output_path: Path where the diff file should be saved

    Returns:
        List of diff lines
    """
    # Read files
    with open(original_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    with open(proofread_path, 'r', encoding='utf-8') as f:
        proofread_lines = f.readlines()

    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        original_lines,
        proofread_lines,
        fromfile=str(original_path.name),
        tofile=str(proofread_path.name),
        lineterm=''
    ))

    # Write diff to file
    if diff_lines:
        with open(diff_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(diff_lines))
            f.write('\n')

    return diff_lines


def generate_html_diff(original_path: Path, proofread_path: Path, html_output_path: Path):
    """
    Generate an HTML side-by-side diff report.

    Args:
        original_path: Path to original HOCR file
        proofread_path: Path to proofread HOCR file
        html_output_path: Path where the HTML file should be saved
    """
    with open(original_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    with open(proofread_path, 'r', encoding='utf-8') as f:
        proofread_lines = f.readlines()

    differ = difflib.HtmlDiff(wrapcolumn=80)
    html = differ.make_file(
        original_lines,
        proofread_lines,
        fromdesc=str(original_path.name),
        todesc=str(proofread_path.name),
        context=True,
        numlines=3
    )

    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def extract_page_elements(hocr_content: str) -> List[Any]:
    """
    Extract all page elements (ocr_page divs) from HOCR content.

    Args:
        hocr_content: HOCR file content as string

    Returns:
        List of BeautifulSoup page div elements
    """
    soup = BeautifulSoup(hocr_content, 'html.parser')
    return soup.find_all('div', class_='ocr_page')


def extract_head(hocr_content: str) -> Optional[Any]:
    """
    Extract the <head> element from HOCR content.

    Args:
        hocr_content: HOCR file content as string

    Returns:
        BeautifulSoup head element or None
    """
    soup = BeautifulSoup(hocr_content, 'html.parser')
    return soup.find('head')


def merge_hocr_files(proofread_files: List[Path], source_hocr: Optional[Path],
                     output_path: Path):
    """
    Merge multiple proofread HOCR files into a single multi-page HOCR file.

    Args:
        proofread_files: List of proofread HOCR file paths (in order)
        source_hocr: Optional path to original source HOCR file (for head content)
        output_path: Path where merged HOCR should be saved
    """
    # Get head content from source or first proofread file
    head_content = None
    if source_hocr and source_hocr.exists():
        with open(source_hocr, 'r', encoding='utf-8') as f:
            head_content = extract_head(f.read())

    if head_content is None and proofread_files:
        with open(proofread_files[0], 'r', encoding='utf-8') as f:
            head_content = extract_head(f.read())

    # Create new HOCR document
    soup = BeautifulSoup('<!DOCTYPE html><html><head></head><body></body></html>', 'html.parser')

    # Add head content
    if head_content:
        soup.head.replace_with(head_content)
    else:
        # Add basic meta tags
        meta_charset = soup.new_tag('meta', charset='utf-8')
        meta_generator = soup.new_tag('meta', content='postproofread.py')
        meta_generator['name'] = 'ocr-system'
        soup.head.append(meta_charset)
        soup.head.append(meta_generator)

    # Collect all page elements from proofread files
    all_pages = []
    for proofread_file in proofread_files:
        with open(proofread_file, 'r', encoding='utf-8') as f:
            content = f.read()
            pages = extract_page_elements(content)
            all_pages.extend(pages)

    # Add all pages to body
    for page in all_pages:
        soup.body.append(page)

    # Write merged HOCR
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    logging.info("✓ Merged %d pages into: %s", len(all_pages), output_path)


def process_batches(input_dir: Path, source_hocr: Optional[Path],
                   output_dir: Path, base_filename: str):
    """
    Process all batch directories to generate diffs and merge HOCR files.

    Args:
        input_dir: Directory containing batch subdirectories
        source_hocr: Optional path to original source HOCR file
        output_dir: Directory for output files
        base_filename: Base name for output files (without extension)
    """
    # Find all batch directories
    batch_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not batch_dirs:
        logging.error("Error: No batch directories found in %s", input_dir)
        sys.exit(1)

    logging.info("Found %d batch directories", len(batch_dirs))

    # Collect all diffs and proofread files
    all_diffs = []
    proofread_files = []
    diff_count = 0

    # FIXME progress bar not showing
    # Use tqdm over batches when available
    iterator = tqdm(batch_dirs, unit='batch') if tqdm else batch_dirs
    for batch_dir in iterator:
        logging.info("Processing batch: %s", batch_dir.name)

        # Find HOCR files in this batch
        original_hocr, proofread_hocr = find_hocr_files(batch_dir)

        if proofread_hocr is None:
            logging.warning("No HOCR files found in %s, skipping", batch_dir.name)
            continue

        proofread_files.append(proofread_hocr)

        # Generate diff if we have both files
        if original_hocr and original_hocr.exists():
            diff_output = batch_dir / f"{proofread_hocr.stem}.diff"

            # Update tqdm description to show which diff is being generated
            try:
                if tqdm and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"Generating {diff_output.name}")
            except Exception:
                pass

            diff_lines = generate_diff(original_hocr, proofread_hocr, diff_output)

            if diff_lines:
                logging.info("Generated diff: %s (%d lines)", diff_output.name, len(diff_lines))
                all_diffs.append(f"=== {batch_dir.name} / {proofread_hocr.name} ===\n")
                all_diffs.extend(diff_lines)
                all_diffs.append("\n")
                diff_count += 1
            else:
                logging.info("No differences found for %s", batch_dir.name)
        else:
            logging.info("No original HOCR file found in %s, skipping diff", batch_dir.name)
            #TODO change to warning?

    # Close tqdm if used
    try:
        if tqdm and hasattr(iterator, 'close'):
            iterator.close()
    except Exception:
        pass

    # Write master diff file
    if all_diffs:
        master_diff_path = output_dir / f"{base_filename}.diff"
        with open(master_diff_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_diffs))
        logging.info("Master diff file: %s (%d batches)", master_diff_path, diff_count)
    else:
        logging.info("No differences found across all batches")

    # Generate HTML diff report
    if all_diffs:
        html_diff_path = output_dir / f"{base_filename}_diff.html"
        generate_html_report(all_diffs, html_diff_path, base_filename)
        logging.info("HTML diff report: %s", html_diff_path)

    # Merge all proofread HOCR files
    if proofread_files:
        merged_output = output_dir / f"{base_filename}-merged.hocr"
        merge_hocr_files(proofread_files, source_hocr, merged_output)
    else:
        logging.warning("No proofread HOCR files found to merge")


def generate_html_report(diff_lines: List[str], output_path: Path, title: str):
    """
    Generate an HTML report from diff text.

    Args:
        diff_lines: List of diff lines
        output_path: Path where HTML should be saved
        title: Title for the HTML report
    """
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Diff Report: {title}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            font-family: Arial, sans-serif;
            color: #333;
        }}
        .diff-container {{
            background-color: white;
            border: 1px solid #ddd;
            padding: 10px;
            overflow-x: auto;
        }}
        .diff-line {{
            white-space: pre;
            padding: 2px 5px;
        }}
        .diff-header {{
            font-weight: bold;
            color: #666;
            background-color: #f0f0f0;
            margin: 10px 0 5px 0;
            padding: 5px;
        }}
        .diff-added {{
            background-color: #d4edda;
            color: #155724;
        }}
        .diff-removed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .diff-context {{
            color: #333;
        }}
        .diff-info {{
            color: #004085;
            background-color: #cce5ff;
        }}
    </style>
</head>
<body>
    <h1>Diff Report: {title}</h1>
    <div class="diff-container">
{content}
    </div>
</body>
</html>
"""

    content_lines = []
    # Use tqdm when available, otherwise plain iterator
    iterator = tqdm(diff_lines, desc="Generating HTML diff", unit="line") if tqdm else diff_lines

    for line in iterator:
        line_str = str(line).rstrip() if not isinstance(line, str) else line.rstrip()

        if line_str.startswith('==='):
            css_class = 'diff-header'
        elif line_str.startswith('+++') or line_str.startswith('---'):
            css_class = 'diff-info'
        elif line_str.startswith('+'):
            css_class = 'diff-added'
        elif line_str.startswith('-'):
            css_class = 'diff-removed'
        elif line_str.startswith('@@'):
            css_class = 'diff-info'
        else:
            css_class = 'diff-context'

        # Escape HTML
        esc_line = (line_str
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;'))

        content_lines.append(f'        <div class="diff-line {css_class}">{esc_line}</div>')

    # Close iterator if it is a tqdm instance
    try:
        if tqdm and hasattr(iterator, 'close'):
            iterator.close()
    except Exception:
        pass

    html = html_template.format(
        title=title,
        content='\n'.join(content_lines)
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # (done) HTML report written


def determine_base_filename(input_dir: Path, source_hocr: Optional[Path]) -> str:
    """
    Determine the base filename for output files.

    Args:
        input_dir: Input directory containing batch subdirectories
        source_hocr: Optional path to original source HOCR file

    Returns:
        Base filename (without extension)
    """
    if source_hocr and source_hocr.exists():
        return source_hocr.stem

    # Try to find first proofread HOCR file
    for batch_dir in sorted(input_dir.iterdir()):
        if batch_dir.is_dir():
            _, proofread = find_hocr_files(batch_dir)
            if proofread:
                # Remove common suffixes
                name = proofread.stem
                for suffix in ['_proofread', '_corrected', '_edited', '_page1', '_page_1']:
                    name = name.replace(suffix, '')
                return name

    # Fallback to input directory name
    return input_dir.name


# Re-add main() which was accidentally removed earlier
def main():
    parser = argparse.ArgumentParser(
        description='Merge proofread HOCR files and generate diff reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python postproofread.py -i ./proofread_batches -s original.hocr
  python postproofread.py -i ./proofread_batches -o ./output
        """
    )

    parser.add_argument(
        '-i', '--input_dir',
        type=Path,
        required=True,
        help='Input directory containing batch subdirectories with proofread HOCR files'
    )

    parser.add_argument(
        '-s', '--source_hocr',
        type=Path,
        help='Optional: Original source HOCR file (before splitting)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=Path,
        help='Output directory (defaults to input directory)'
    )

    parser.add_argument(
        '-v', '--verbosity',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Logging verbosity for terminal output (default: WARNING)'
    )

    args = parser.parse_args()

    # Configure logging according to verbosity flag (default WARNING)
    log_level = getattr(logging, args.verbosity.upper(), logging.WARNING)
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    # Validate input directory
    if not args.input_dir.exists():
        logging.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    if not args.input_dir.is_dir():
        logging.error("Input path is not a directory: %s", args.input_dir)
        sys.exit(1)

    # Validate source HOCR if provided
    if args.source_hocr and not args.source_hocr.exists():
        logging.warning("Source HOCR file does not exist: %s", args.source_hocr)
        args.source_hocr = None

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine base filename
    base_filename = determine_base_filename(args.input_dir, args.source_hocr)

    logging.info("Input directory: %s", args.input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Base filename: %s", base_filename)
    if args.source_hocr:
        logging.info("Source HOCR: %s", args.source_hocr)

    # Process all batches
    process_batches(args.input_dir, args.source_hocr, output_dir, base_filename)

    logging.info("\n✓ Processing complete!")


if __name__ == '__main__':
    main()
