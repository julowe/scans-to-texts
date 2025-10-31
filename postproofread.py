#!/usr/bin/env python3
"""
postproofread.py - Merge proofread HOCR files and generate diff reports

This script processes the output of the proofreading workflow by:
1. Comparing original and proofread HOCR files to generate diffs
2. Merging proofread HOCR files back into a single multi-page HOCR file
3. Generating diff reports (text and HTML) showing all changes
4. Optionally generating a LaTeX formatted file with font preservation

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


def extract_bbox(title_attr: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box coordinates from HOCR title attribute.

    Args:
        title_attr: Title attribute string (e.g., "bbox 100 200 300 400")

    Returns:
        Tuple of (x0, y0, x1, y1) or None if not found
    """
    if not title_attr:
        return None
    match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title_attr)
    if match:
        coords = tuple(map(int, match.groups()))
        return coords  # type: ignore
    return None


def detect_font_style(word_span) -> Tuple[bool, bool, bool]:
    """
    Detect font styling from HOCR word span.

    Args:
        word_span: BeautifulSoup span element

    Returns:
        Tuple of (is_bold, is_italic, is_superscript)
    """
    title = word_span.get('title', '')
    style = word_span.get('style', '')

    # Check for bold
    is_bold = 'bold' in title.lower() or 'bold' in style.lower()

    # Check for italic
    is_italic = 'italic' in title.lower() or 'italic' in style.lower()

    # Check for superscript (common in footnote markers)
    is_superscript = 'vertical-align:super' in style or 'superscript' in title.lower()
    # TODO also detect superscript by comparing y position of bounding box to other boxes (and/or before after?)

    return is_bold, is_italic, is_superscript


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for LaTeX
    """
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }

    # TODO: convert quotes and doublequotes here? or better elsewhere?
    # TODO: for arabic: check on hamza vs ayn marks in middle of words? (harder for ones at beginning?)

    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)

    return result


def format_latex_text(text: str, is_bold: bool, is_italic: bool, is_superscript: bool) -> str:
    """
    Format text with LaTeX commands based on font style.

    Args:
        text: Text to format
        is_bold: Whether text should be bold
        is_italic: Whether text should be italic
        is_superscript: Whether text should be superscript

    Returns:
        LaTeX formatted text
    """
    escaped = escape_latex(text)

    if is_superscript:
        escaped = f'\\textsuperscript{{{escaped}}}'

    if is_bold and is_italic:
        escaped = f'\\textbf{{\\textit{{{escaped}}}}}'
    elif is_bold:
        escaped = f'\\textbf{{{escaped}}}'
    elif is_italic:
        escaped = f'\\textit{{{escaped}}}'

    return escaped


def is_new_paragraph(prev_bbox: Optional[Tuple[int, int, int, int]],
                     curr_bbox: Optional[Tuple[int, int, int, int]],
                     threshold_factor: float = 1.5) -> bool:
    """
    Determine if the current line starts a new paragraph based on vertical spacing.

    Args:
        prev_bbox: Previous line bounding box (x0, y0, x1, y1)
        curr_bbox: Current line bounding box (x0, y0, x1, y1)
        threshold_factor: Multiplier for typical line height to detect paragraph breaks

    Returns:
        True if the current line should start a new paragraph
    """
    if not prev_bbox or not curr_bbox:
        return False

    prev_height = prev_bbox[3] - prev_bbox[1]
    vertical_gap = curr_bbox[1] - prev_bbox[3]

    # TODO also check for indents? might not be much line space...
    # Yeah... no line space in current text for new paragraph, only indents...
    # get mode line start x position and count anything in side that as a new paragraph,
    # and anything outside that as a margin line note???
    # NB: margin notes will change 'sides' relative to paragraph indents according to
    # even pages (left side notes) vs odd pages (right side notes)

    # New paragraph if gap is significantly larger than typical line height
    return vertical_gap > (prev_height * threshold_factor)


def detect_footnotes(page_soup) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
    """
    Detect footnotes in a page (typically at bottom, may be in columns).

    Args:
        page_soup: BeautifulSoup element for ocr_page div

    Returns:
        List of (footnote_number, footnote_text, bbox) tuples
    """
    footnotes = []

    # Get all lines in page
    all_lines = page_soup.find_all('span', class_='ocr_line')
    if not all_lines:
        return footnotes

    # Get page height to identify bottom region
    page_bbox = extract_bbox(page_soup.get('title', ''))
    if not page_bbox:
        return footnotes

    page_height = page_bbox[3]
    bottom_threshold = page_height * 0.75  # Bottom 25% of page

    # TODO also consider font size (usually smaller)
    # NB: 0294.jpg has one large left aligned footnote that is full width of page,
    # and then a ~10 character footnote that is right aligned after the prev footnote,
    # but on the same line as the prev footnote... ugh.

    # Look for lines in bottom region that start with numbers or superscript
    for line in all_lines:
        line_bbox = extract_bbox(line.get('title', ''))
        if not line_bbox or line_bbox[1] < bottom_threshold:
            continue

        # Get line text
        line_text = line.get_text(strip=True)

        # Check if starts with number (footnote marker)
        match = re.match(r'^(\d+)[.\s]', line_text)
        if match:
            footnote_num = match.group(1)
            footnote_text = line_text[len(match.group(0)):].strip()
            footnotes.append((footnote_num, footnote_text, line_bbox))

    return footnotes


def generate_latex_document(merged_hocr_path: Path, output_path: Path, enable_page_breaks: bool = False):
    """
    Generate a LaTeX document from merged HOCR file.

    Args:
        merged_hocr_path: Path to merged HOCR file
        output_path: Path where LaTeX file should be saved
        enable_page_breaks: Whether to include \\clearpage between pages
    """
    with open(merged_hocr_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # TODO make real
    enable_line_breaks = True

    # TODO make book not article. see how much that messes things up TEST
    # Start LaTeX document
    latex_lines = [
        r'\documentclass[12pt]{article}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{geometry}',
        r'\geometry{letterpaper, margin=1in}',
        r'\usepackage{parskip}',
        r'',
        r'% Logic to optionally enable commands to retain page breaks and/or retain line breaks',
        r'\newbool{retainpagebreaks} % creates boolean variable and initializes it to `false`',
        r'\newbool{retainlinebreaks} % creates boolean variable and initializes it to `false`',
    ]

    if enable_page_breaks:
        latex_lines.append(r'\booltrue{retainpagebreaks} % comment (i.e. add a leading %) to NOT set it to `true`')
    else:
        latex_lines.append(r'%\booltrue{retainpagebreaks} % uncomment (i.e. delete the leading %) to set it to `true`')

    if enable_line_breaks:
        latex_lines.append(r'\booltrue{retainlinebreaks} % comment (i.e. add a leading %) to NOT set it to `true`')
    else:
        latex_lines.append(r'%\booltrue{retainlinebreaks} % uncomment (i.e. delete the leading %) to set it to `true`')

    latex_lines.extend([
        r'',
        r'\ifbool{retainpagebreaks}',
        r'    {\newcommand{\optionalpagebreak}{\clearpage}}',
        r'    {\newcommand{\optionalpagebreak}{}}',
        r'',
        r'\ifbool{retainlinebreaks}',
        r'    {\newcommand{\optionallinebreak}{\\}}',
        r'    {\newcommand{\optionallinebreak}{}}',
        r'',
        r'\begin{document}',
        r'',
    ])

    # Process each page
    pages = soup.find_all('div', class_='ocr_page')

    for page_idx, page in enumerate(pages):
        if page_idx > 0:
            latex_lines.append(r'\optionalpagebreak')
            latex_lines.append(r'')

        # Add page comment if present
        page_title = page.get('title', '')
        if 'image' in page_title or 'ppageno' in page_title:
            # Extract overall page comment from HTML comments
            for comment in page.find_all(string=lambda text: isinstance(text, str) and '<!-- overall page' in text):
                latex_lines.append(f'% {comment.strip()}')

        # Look for page number or ID in title
        page_match = re.search(r'ppageno\s+(\d+)', page_title)
        if page_match:
            latex_lines.append(f'% Page {page_match.group(1)}')

        latex_lines.append(r'')

        # Detect footnotes for this page
        footnotes = detect_footnotes(page)
        footnote_dict = {num: text for num, text, bbox in footnotes}
        footnote_bboxes = {num: bbox for num, text, bbox in footnotes}

        # Process lines
        prev_line_bbox = None

        for line in page.find_all('span', class_='ocr_line'):
            line_bbox = extract_bbox(line.get('title', ''))

            # Check if this is a footnote line (skip in main text)
            if footnote_bboxes:
                is_footnote_line = any(
                    line_bbox and fb and abs(line_bbox[1] - fb[1]) < 10
                    for fb in footnote_bboxes.values()
                )
                if is_footnote_line:
                    continue

            # Detect paragraph breaks
            if prev_line_bbox and line_bbox and is_new_paragraph(prev_line_bbox, line_bbox):
                latex_lines.append(r'')  # Blank line for new paragraph

            # Process words in line
            line_text_parts = []
            words = line.find_all('span', class_='ocrx_word')

            for word in words:
                word_text = word.get_text(strip=True)
                if not word_text:
                    continue

                # Detect font styling
                is_bold, is_italic, is_superscript = detect_font_style(word)

                # Check if this is a footnote reference
                if is_superscript and word_text.isdigit() and word_text in footnote_dict:
                    # Add footnote reference
                    formatted_word = f'\\footnote{{{escape_latex(footnote_dict[word_text])}}}'
                else:
                    formatted_word = format_latex_text(word_text, is_bold, is_italic, is_superscript)

                line_text_parts.append(formatted_word)

            if line_text_parts:
                latex_lines.append(' '.join(line_text_parts) + ' %end-of-line')

            prev_line_bbox = line_bbox

        # Add footnotes as comments at bottom of page
        if footnotes:
            latex_lines.append(r'')
            latex_lines.append(r'% Footnotes for this page:')
            for num, text, bbox in footnotes:
                latex_lines.append(f'% [{num}] {text}')

    # End document
    latex_lines.extend([
        r'',
        r'\end{document}'
    ])

    # Write LaTeX file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))

    logging.info("✓ Generated LaTeX document: %s", output_path)


def process_batches(input_dir: Path, source_hocr: Optional[Path],
                   output_dir: Path, base_filename: str, generate_latex: bool = False,
                   latex_page_breaks: bool = False):
    """
    Process all batch directories to generate diffs and merge HOCR files.

    Args:
        input_dir: Directory containing batch subdirectories
        source_hocr: Optional path to original source HOCR file
        output_dir: Directory for output files
        base_filename: Base name for output files (without extension)
        generate_latex: Whether to generate LaTeX output
        latex_page_breaks: Whether to enable page breaks in LaTeX output
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

        # Generate LaTeX if requested
        if generate_latex:
            latex_output = output_dir / f"{base_filename}.tex"
            generate_latex_document(merged_output, latex_output, enable_page_breaks=latex_page_breaks)
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
  python postproofread.py -i ./proofread_batches --latex --latex-page-breaks
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

    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX formatted output file'
    )

    parser.add_argument(
        '--latex-page-breaks',
        action='store_true',
        help='Enable page breaks (\\clearpage) in LaTeX output between pages'
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
    process_batches(args.input_dir, args.source_hocr, output_dir, base_filename,
                   generate_latex=args.latex, latex_page_breaks=args.latex_page_breaks)

    logging.info("\n✓ Processing complete!")


if __name__ == '__main__':
    main()
