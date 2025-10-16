# coding: utf-8

# -------------------------------------------------------------------------
# Additions by Justin Lowe.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# Retrieved from: https://github.com/Azure-Samples/document-intelligence-code-samples/blob/8ac994c2712f3104d00ca4224b0ded6fd1ecbf70/Python(v4.0)/Read_model/sample_analyze_read.py

"""
FILE: sample_analyze_read.py

DESCRIPTION:
    This sample demonstrates how to extract document information using "prebuilt-read"
    to analyze a given file.

PREREQUISITES:
    The following prerequisites are necessary to run the code. For more details, please visit the "How-to guides" link: https://aka.ms/how-to-guide

    -------Python and IDE------
    1) Install Python 3.8 or later (https://www.python.org/), which should include pip (https://pip.pypa.io/en/stable/).
    2) Install the latest version of Visual Studio Code (https://code.visualstudio.com/) or your preferred IDE.

    ------Azure AI services or Document Intelligence resource------
    Create a single-service (https://aka.ms/single-service) or multi-service (https://aka.ms/multi-service) resource.
    You can use the free pricing tier (F0) to try the service and upgrade to a paid tier for production later.

    ------Get the key and endpoint------
    1) After your resource is deployed, select "Go to resource".
    2) In the left navigation menu, select "Keys and Endpoint".
    3) Copy one of the keys and the Endpoint for use in this sample.

    ------Set your environment variables------
    At a command prompt, run the following commands, replacing <yourKey> and <yourEndpoint> with the values from your resource in the Azure portal.
    1) For Windows:
       setx DOCUMENTINTELLIGENCE_API_KEY <yourKey>
       setx DOCUMENTINTELLIGENCE_ENDPOINT <yourEndpoint>
       • You need to restart any running programs that read the environment variable.
    2) For macOS:
       export DOCUMENTINTELLIGENCE_API_KEY=<yourKey>
       export DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
       • This is a temporary environment variable setting method that only lasts until you close the terminal session.
       • To set an environment variable permanently, visit: https://aka.ms/set-environment-variables-for-macOS
    3) For Linux:
       export DOCUMENTINTELLIGENCE_API_KEY=<yourKey>
       export DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
       • This is a temporary environment variable setting method that only lasts until you close the terminal session.
       • To set an environment variable permanently, visit: https://aka.ms/set-environment-variables-for-Linux

    ------Set up your programming environment------
    At a command prompt, run the following code to install the Azure AI Document Intelligence client library for Python with pip:
    pip install azure-ai-documentintelligence

    ------Create your Python application------
    1) Create a new Python file called "sample_analyze_read.py" in an editor or IDE.
    2) Open the "sample_analyze_read.py" file and insert the provided code sample into your application.
    3) At a command prompt, use the following command to run the Python file:
       python sample_analyze_read.py
"""

import argparse
import glob
import os
from datetime import datetime

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential


def format_bounding_box(bounding_box):
    if not bounding_box:
        return "N/A"
    return "[{}, {}], [{}, {}], [{}, {}], [{}, {}]".format(
        bounding_box[0],
        bounding_box[1],
        bounding_box[2],
        bounding_box[3],
        bounding_box[4],
        bounding_box[5],
        bounding_box[6],
        bounding_box[7],
    )


def analyze_read(
        file_pattern,
        print_page_dimensions=False,
):
    # For how to obtain the endpoint and key, please see PREREQUISITES above.
    # Values are expected to be provided via environment variables or a .env file:
    #   DOCUMENTINTELLIGENCE_ENDPOINT
    #   DOCUMENTINTELLIGENCE_API_KEY
    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

    if not endpoint or not key:
        raise EnvironmentError(
            "Missing DOCUMENTINTELLIGENCE_ENDPOINT or DOCUMENTINTELLIGENCE_API_KEY. "
            "Create a .env file with these variables or set them in your environment."
        )

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    # Get all files matching the pattern
    file_paths = sorted(glob.glob(file_pattern, recursive=True))

    if not file_paths:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(file_paths)} files to process")

    # Generate markdown output for all files
    markdown_content = ["# OCR Results\n\n"]

    # Generate hOCR output
    hocr_content = ['<?xml version="1.0" encoding="UTF-8"?>\n',
                    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n',
                    '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n', '<head>\n',
                    '<title>OCR Results</title>\n',
                    '<meta http-equiv="content-type" content="text/html; charset=utf-8" />\n',
                    '<meta name="ocr-system" content="Azure Document Intelligence" />\n',
                    '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />\n',
                    '</head>\n', '<body>\n']

    # Generate CSV output
    csv_content = ["filename,word,confidence,left_x,upper_y\n"]

    page_counter = 0

    for file_idx, file_path in enumerate(file_paths):
        print(
            f"Processing file {file_idx + 1}/{len(file_paths)}: {os.path.basename(file_path)}"
        )

        markdown_content.append(f"## Filename: {os.path.basename(file_path)}\n\n")
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-read",
                body=f,
                content_type="application/octet-stream",
                query_fields=None,
                output_content_format="markdown"
            )
        result = poller.result()

        pages_in_file = len(result.pages) if result.pages else 0

        for page in result.pages:
            page_counter += 1

            if pages_in_file > 1:
                markdown_content.append("### Page {}\n\n".format(page_counter))

            # hOCR page
            hocr_content.append(
                f'<div class="ocr_page" id="page_{page_counter}" title="image {os.path.basename(file_path)}; bbox 0 0 {int(page.width)} {int(page.height)}; ppageno {page_counter}">\n')
            hocr_content.append(f'\t<div class="ocr_carea" id="carea_{page_counter}_1">\n')

            if print_page_dimensions:
                markdown_content.append(
                    f"*Page dimensions: {page.width} x {page.height} {page.unit}*\n\n"
                )

            # Group lines into paragraphs based on vertical spacing
            if page.lines:
                current_paragraph = []
                prev_y = None
                para_counter = 0

                for line_idx, line in enumerate(page.lines):
                    # Get the y-coordinate (top of bounding box)
                    if line.polygon and len(line.polygon) >= 2:
                        current_y = line.polygon[1]

                        # If there's a significant vertical gap, start a new paragraph
                        if prev_y is not None and abs(current_y - prev_y) > 20:
                            if current_paragraph:
                                markdown_content.append(
                                    " ".join(current_paragraph) + "\n\n"
                                )
                                current_paragraph = []
                                para_counter += 1

                        current_paragraph.append(line.content)
                        prev_y = line.polygon[7]  # Bottom y-coordinate

                        # hOCR line
                        bbox = f"{int(line.polygon[0])} {int(line.polygon[1])} {int(line.polygon[4])} {int(line.polygon[5])}"
                        hocr_content.append(
                            f'\t<span class="ocr_line" id="line_{page_counter}_{line_idx}" title="bbox {bbox}">\n')

                        # hOCR words
                        if page.words:
                            for word_idx, word in enumerate(page.words):
                                # Check if the word belongs to this line (simple check by y-coordinate proximity)
                                if word.polygon and abs(word.polygon[1] - line.polygon[1]) < 5:
                                    word_bbox = f"{int(word.polygon[0])} {int(word.polygon[1])} {int(word.polygon[4])} {int(word.polygon[5])}"
                                    confidence = int(word.confidence * 100) if word.confidence else 0
                                    hocr_content.append(
                                        f'\t\t<span class="ocrx_word" id="word_{page_counter}_{line_idx}_{word_idx}" title="bbox {word_bbox}; x_wconf {confidence}">{word.content}</span>\n')

                        hocr_content.append('\t</span>\n\n')  # Close ocr_line
                    else:
                        current_paragraph.append(line.content)

                # Add the last paragraph
                if current_paragraph:
                    markdown_content.append(" ".join(current_paragraph) + "\n\n")

            # Process words for CSV output
            if page.words:
                for word in page.words:
                    if word.polygon and len(word.polygon) >= 2:
                        word_text = word.content
                        confidence = f"{word.confidence:.4f}" if word.confidence else "0.0000"
                        left_x = int(word.polygon[0])
                        upper_y = int(word.polygon[1])
                        # Quote the word column and escape any internal quotes by doubling them
                        safe_word = (word_text or "").replace('"', '""')

                        csv_content.append(f'{filename},"{safe_word}",{confidence},{left_x},{upper_y}\n')

            hocr_content.append('\t</div>\n')  # Close ocr_carea
            hocr_content.append('</div>\n')  # Close ocr_page

        # Print console output for this file
        print(
            f"Document {file_idx + 1} contains content: ", result.content[:100] + "..."
        )

    # Close hOCR
    hocr_content.append('</body>\n')
    hocr_content.append('</html>\n')

    # Write all results to files
    output_dir = os.path.dirname(file_paths[0])

    # Build timestamped base filename: doc-int-results-YYYY-MM-DDTHH:MM
    timestamp = datetime.now().isoformat(timespec='minutes')
    # Sanitize for Windows filenames (colons are invalid)
    if os.name == 'nt':
        timestamp = timestamp.replace(':', '-')
    output_base_name = f"doc-int-results-{timestamp}"

    # Markdown output
    output_path_md = os.path.join(output_dir, f"{output_base_name}.md")
    with open(output_path_md, "w", encoding="utf-8") as f:
        f.writelines(markdown_content)
    print(f"\nMarkdown file with all results saved to: {output_path_md}")

    # hOCR output
    output_path_hocr = os.path.join(output_dir, f"{output_base_name}.hocr")
    with open(output_path_hocr, "w", encoding="utf-8") as f:
        f.writelines(hocr_content)
    print(f"hOCR file with all results saved to: {output_path_hocr}")

    # CSV output
    output_path_csv = os.path.join(output_dir, f"{output_base_name}.csv")
    with open(output_path_csv, "w", encoding="utf-8") as f:
        f.writelines(csv_content)
    print(f"CSV file with all results saved to: {output_path_csv}")

    print("----------------------------------------")


if __name__ == "__main__":
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    parser = argparse.ArgumentParser(
        description=(
            "Analyze documents (images or PDFs) with Azure Document Intelligence "
            "using a bash-style glob pattern."
        )
    )
    parser.add_argument(
        "-p", "--pattern", dest="file_pattern", metavar="GLOB", required=True,
        help=(
            "Bash-style glob pattern for input files, e.g. '/path/to/files/*.jpg' "
            "or '/path/**/*.pdf'. Use quotes to prevent shell pre-expansion."
        ),
    )
    parser.add_argument(
        "--print-page-dimensions", action="store_true",
        help="Print page dimensions for each page.",
    )
    args = parser.parse_args()

    try:
        load_dotenv(find_dotenv())
        analyze_read(
            file_pattern=args.file_pattern,
            print_page_dimensions=args.print_page_dimensions,
        )
    except HttpResponseError as error:
        if error.error is not None:
            print(f"Received service error: {error.error}")
            raise
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Invalid request: {error}")
        raise
