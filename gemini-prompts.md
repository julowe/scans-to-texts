# Prompts for Gemini OCR Requests

## My Iterations

Descriptions in plain text, prompts inside backticks.

### Current Proofreading Prompt

Used 2025-10-25:

```python
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
```

#### Prior Attempts

Descriiption later?

```python 
    "Rules:\n"
"- Preserve the hOCR structure and tags exactly (including class names, bbox attributes, and hierarchy).\n"
"- Only correct textual content inside text nodes. Do not remove or add HTML elements.\n"
"- Do not invent content; only correct what is clearly wrong per the images.\n"
"- Do not include any explanations. Respond ONLY with the corrected hOCR HTML.\n\n"
"Draft to correct (hOCR):\n\n" + draft_text
```

```python 
    # WORKED! got text words back.
"You are an expert proofreader for OCR outputs. Given the hOCR HTML document to correct that is copied"
" below, please transcribe back just the 'text' objects from this hocr document. remove all the xml and "
"just give me a plain text message "
"of what the scanned page says.\n\n"
"hOCR HTML document to correct:\n\n" + draft_text
```

```python 
    # FAILED, no text back
"You are an expert proofreader for OCR outputs. Given an hOCR HTML document and reference images "
"that contain the true source pages, correct mistakes strictly based on the images.\n\n"
"Rules:\n"
"- Preserve the hOCR structure and tags exactly (including class names, bbox attributes, and hierarchy).\n"
"- Only correct textual content inside text nodes. Do not remove or add HTML elements.\n"
"- Do not invent content; only correct what is clearly wrong per the images.\n"
"- Do not include any explanations. Respond ONLY with the corrected hOCR HTML.\n\n"
"Draft to correct (hOCR):\n\n" + draft_text
```

```python 
    # FAILED, no text back
"You are an expert proofreader for OCR outputs. Given an hOCR HTML document and reference images "
"that contain the true source pages, correct mistakes strictly based on the images.\n\n"
"Rules:\n"
"- Preserve the hOCR structure and tags exactly (including class names, bbox attributes, and hierarchy).\n"
"- Only correct textual content inside text nodes. Do not remove or add HTML elements.\n"
"- Do not invent content; only correct what is clearly wrong per the images.\n"
"- Keep whitespace and line breaks consistent with the original where possible.\n"
"- Do not include any explanations. Respond ONLY with the corrected hOCR HTML.\n\n"
"Draft to correct (hOCR):\n\n" + draft_text
```

### Markdown Proofreading Prompt

Default, I did not change at all, just copying here for record keeping.

```python
    "You are an expert proofreader for OCR outputs. Given a draft in Markdown and reference images "
"that contain the true source pages, correct mistakes in the draft strictly based on the images.\n\n"
"Rules:\n"
"- Preserve Markdown structure (headings, lists, emphasis).\n"
"- Do not invent content; only correct what is clearly wrong per the images.\n"
"- Keep paragraph breaks and footnotes, preserving superscripts/subscripts where applicable.\n"
"- Maintain any explicit page or section markers if present.\n"
"- Respond ONLY with the corrected Markdown, no explanations.\n\n"
"Draft to correct (Markdown):\n\n" + draft_text
```

## OCR Prompts

For doing OCR on images with no other reference.

TODO: split up prompts and describe pros and cons of each.

```python
DEFAULT_PROMPT = (
    "Act like a text scanner and transcribe the text in the image."
    "Extract text as it is without analyzing it and without summarizing it. Treat "
    "all images as a whole document and analyze them accordingly. Think of it as a document with multiple "
    "pages, each image being a page. Understand page-to-page flow logically and semantically."
    "Please mark the beginning and end of each page clearly in your response."
    "Please print the page number according to the overall files processed"
    "(do not print the total file number),"
    "as well as the page number extracted from the current image."
    "Please keep track of the chapters and sections in your response."
    "Please put a header before the footnotes at the bottom of the page of `### FOOTNOTES`"
)

prompt_1 = (
    "Act like a text scanner and transcribe the text in the image."
    "Extract text as it is without analyzing it and without summarizing it. Treat "
    "all images as a whole document and analyze them accordingly. Think of it as a document with multiple "
    "pages, each image being a page. Understand page-to-page flow logically and semantically."
    "Please mark the beginning and end of each page clearly in your response."
    "Please keep track of the chapters and sections in your response."
)

# NOTE: THIS FAILED, no text returned twice!
prompt_1_2 = (
    "Act like a text scanner and transcribe the text in the image."
    "Extract text as it is without analyzing it and without summarizing it. Treat "
    "all images as a whole document and analyze them accordingly. Think of it as a document with multiple "
    "pages, each image being a page. Understand page-to-page flow logically and semantically."
    "Please mark the beginning and end of each page clearly in your response."
    "Please print the page number according to the overall files processed"
    "(do not print the total file number),"
    "as well as the page number extracted from the current image."
    "Please keep track of the chapters and sections in your response."
    "Please put a header before the footnotes at the bottom of the page of `### FOOTNOTES -PAGENUMBER-`"
    "where -PAGENUMBER- is the number of the file currently being processed."
)

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
prompt_google_simple = (
    "Analyze these images in detail. Describe the objects, people, setting, colors, and any notable "
    "features. If there is text visible, please transcribe it."
)

prompt_2 = (
    "Act like a text scanner and transcribe the text in the image. "
    "Please make sure to transcribe the characters with their correct diacritics and spacing. "
    "Treat images as part of one book and analyze them accordingly. "
    "Think of it as a document with multiple pages, each image being a page. "
    "Understand page-to-page flow logically and semantically. "
    "Please mark the beginning and end of each page clearly in your response. "
    "Please keep track of the chapters and sections in your response. "
)

prompt_3 = (
    "Act like a text scanner and transcribe the text in the image."
    "Please make sure to transcribe the characters with their correct diacritics and spacing."
    "Treat images as part of one book and analyze them accordingly. "
    "Think of it as a document with multiple pages, each image being a page. "
    "Understand page-to-page flow logically and semantically."
    "Please mark the beginning and end of each page clearly in your response."
    "Please keep track of the chapters and sections in your response."
    "Correct OCR-induced errors in the text, following these guidelines:\n"
    "1. Fix OCR-induced typos and errors:\n"
    "   - Correct words split across line breaks\n"
    "   - Fix common OCR errors (e.g., 'rn' misread as 'm')\n"
    "   - Use context and common sense to correct errors\n"
    "   - Only fix clear errors, don't alter the content unnecessarily\n"
    "   - Do not add extra periods or any unnecessary punctuation\n"
    "2. Maintain original structure:\n"
    "   - Keep all headings and subheadings intact\n"
    "3. Preserve original content:\n"
    "   - Keep all important information from the original text\n"
    "   - Do not add any new information not present in the original text\n"
    "   - Remove unnecessary line breaks within sentences or paragraphs\n"
    "   - Maintain paragraph breaks\n"
    "4. Maintain coherence:\n"
    "   - Ensure the content connects smoothly with the previous context\n"
    "   - Handle text that starts or ends mid-sentence appropriately\n"
    "IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. "
    "Do not include any introduction, explanation, or metadata."
)

prompt_4 = (
    "Act like a text scanner and transcribe the text in the image."
    "Please make sure to transcribe the characters with their correct diacritics and spacing."
    "Treat images as part of one book and analyze them accordingly. "
    "Think of it as a document with multiple pages, each image being a page. "
    "Understand page-to-page flow logically and semantically."
    "Please mark the beginning and end of each page clearly in your response."
    "Please keep track of the chapters and sections in your response."
    "Correct OCR-induced errors in the text, following these guidelines:\n"
    "1. Fix OCR-induced typos and errors:\n"
    "   - Correct words split across line breaks\n"
    "   - Fix common OCR errors (e.g., 'rn' misread as 'm')\n"
    "   - Use context and common sense to correct errors involving one or two characters\n"
    "   - Only fix clear errors, don't alter the content more than 3 characters in one place\n"
    "   - Do not add extra periods or any unnecessary punctuation\n"
    "2. Maintain original structure:\n"
    "   - Keep all headings and subheadings intact\n"
    "3. Preserve original content:\n"
    "   - Keep all important information from the original text\n"
    "   - Do not add any new information not present in the original text\n"
    "   - Remove unnecessary line breaks within sentences or paragraphs\n"
    "   - Maintain paragraph breaks\n"
    "IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. "
    "Do not include any introduction, explanation, or metadata."
)

prompt_5 = (
    "Act like a text scanner and transcribe the text in the image."
    "Please make sure to transcribe the characters with their correct diacritics and spacing."
    "Treat images as part of one book and analyze them accordingly. "
    "Think of it as a document with multiple pages, each image being a page. "
    "Understand page-to-page flow logically and semantically."
    "Correct OCR-induced errors in the text, following these guidelines:\n"
    "1. Fix OCR-induced typos and errors:\n"
    "   - Correct words split across line breaks\n"
    "   - Fix common OCR errors (e.g., 'rn' misread as 'm')\n"
    "   - Use context and common sense to correct errors involving only one or two characters\n"
    "   - Only fix clear errors in one or two characters, don't alter the content of more than 3 characters in one place\n"
    "   - Do not add extra periods or any unnecessary punctuation\n"
    "2. Maintain original structure:\n"
    "   - Keep all headings and subheadings intact\n"
    "3. Preserve original content:\n"
    "   - Keep all important information from the original text\n"
    "   - Do not add any new information not present in the original text\n"
    "   - Remove unnecessary line breaks within sentences or paragraphs\n"
    "   - Maintain paragraph breaks\n"
    "IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks.\n"
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
    "Please transcribe all superscripts and subscripts and also footnotes."
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

## Usage Notes
# Prompt 3 inserted more diacritics incorrectly e.g. Medīna, caught some footnotes, missed others.
# Prompt 4 removed some context etc instructions
# Prompt 5 DIDNT WORK NO TEXT. added page numbers and part numbers. margin notes does not work, screws up footnotes.

# prompt_to_use = prompt_1_3

```

## Collected from web/repos

### Step 1: OCR Correction

ocr_correction_prompt = f"""
Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
    - Correct words split across line breaks
    - Fix common OCR errors (e.g., 'rn' misread as 'm')
    - Use context and common sense to correct errors
    - Only fix clear errors, don't alter the content unnecessarily
    - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
    - Keep all headings and subheadings intact

3. Preserve original content:
    - Keep all important information from the original text
    - Do not add any new information not present in the original text
    - Remove unnecessary line breaks within sentences or paragraphs
    - Maintain paragraph breaks

4. Maintain coherence:
    - Ensure the content connects smoothly with the previous context
    - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include
any introduction, explanation, or metadata.

Previous context:
{prev_context[-500:]}

Current chunk to process:
{chunk}

Corrected text:
"""

### Step 2: Markdown Formatting (if requested)

markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original
structure. Follow these guidelines:

1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for
   subtitles, etc.)
    - Ensure each heading is on its own line
    - Add a blank line before and after each heading
2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for
   example, "cor- rect" should be "correct")
3. Format lists properly (unordered or ordered) if they exist in the original text
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
5. Preserve all original content and meaning
6. Do not add any extra punctuation or modify the existing punctuation
7. Remove any spuriously inserted introductory text such as "Here is the corrected text:" that may have been added by
   the LLM and which is obviously not part of the original text.
8. Remove any obviously duplicated content that appears to have been accidentally included twice. Follow these strict
   guidelines:
    - Remove only exact or near-exact repeated paragraphs or sections within the main chunk.
    - Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries.
    - Do not remove content that is simply similar but conveys different information.
    - Preserve all unique content, even if it seems redundant.
    - Ensure the text flows smoothly after removal.
    - Do not add any new content or explanations.
    - If no obvious duplicates are found, return the main chunk unchanged.
9. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as
   blockquotes." if not suppress_headers_and_page_numbers else "Carefully remove headers, footers, and page numbers
   while preserving all other content."}

Text to reformat:

{ocr_corrected_chunk}

Reformatted markdown:
"""

### Code for small python script

```python
https: // apidog.com / blog / gemini - 2 - 0 - flash - ocr /
from PIL import Image
# from IPython.display import display
# import base64
import io


def ocr_with_gemini(image_paths, instruction):
    """Process images with Gemini 2.0 Flash for OCR"""
    images = [Image.open(path) for path in image_paths]

    prompt = f"""
    {instruction}

    These are pages from a PDF document. Extract all text content while preserving the structure.
    Pay special attention to tables, columns, headers, and any structured content.
    Maintain paragraph breaks and formatting.
    """

    response = model.generate_content([prompt, *images])
    return response.text
```
