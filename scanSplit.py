"""
PDF Dual Page Splitter & Text Enhancer
Converts dual-page PDFs to single-page PDFs with optional text enhancement for OCR.
Handles both evenly split pages and angled scans with shadows.

USAGE EXAMPLES:
    # Just split pages (no enhancement) - RECOMMENDED DEFAULT
    python script.py "book.pdf" --no-enhance

    # Split all PDFs in current directory (no enhancement)
    python script.py --no-enhance

    # Split with text enhancement (for poor quality scans only)
    python script.py "book.pdf"

    # Process multiple files without enhancement
    python script.py *.pdf --no-enhance

    # High quality scan processing
    python script.py "book.pdf" --dpi 600 --no-enhance

OUTPUT:
    - Creates scan_fix_output/ directory
    - Generates filename_split.pdf for each input
    - Left page first, then right page (proper reading order)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging
from PIL import Image
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFPageSplitter:
    def __init__(self, shadow_threshold: int = 50, min_shadow_height: float = 0.3, dpi: int = 300):
        """
        Initialize the PDF page splitter.

        Args:
            shadow_threshold: Darkness threshold for detecting shadows (0-255)
            min_shadow_height: Minimum height ratio for valid shadow detection
            dpi: DPI for PDF rendering
        """
        self.shadow_threshold = shadow_threshold
        self.min_shadow_height = min_shadow_height
        self.dpi = dpi

    def enhance_text(self, image: np.ndarray) -> np.ndarray:
        """
        Gently enhance text quality for better OCR results.
        Conservative approach to avoid artifacts.

        Args:
            image: Input image as numpy array

        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Very gentle CLAHE to improve contrast slightly
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)

        # Light denoising only
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # Very subtle sharpening
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Ensure we don't go beyond original bounds
        result = np.clip(sharpened, 0, 255).astype(np.uint8)

        return result

    def find_shadow_split(self, image: np.ndarray) -> Optional[int]:
        """
        Find the center split line by detecting page shadows.

        Args:
            image: Input image as numpy array

        Returns:
            X coordinate of split line, or None if not found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        center_region = width // 4  # Search in center 50% of image
        start_x = width // 2 - center_region // 2
        end_x = width // 2 + center_region // 2

        # Calculate vertical intensity profile in center region
        intensity_profile = []
        for x in range(start_x, end_x):
            column = gray[:, x]
            avg_intensity = np.mean(column)
            intensity_profile.append(avg_intensity)

        # Find the darkest region (shadow)
        min_intensity = min(intensity_profile)
        if min_intensity > self.shadow_threshold:
            return None  # No shadow detected

        # Find consecutive dark pixels
        dark_indices = []
        for i, intensity in enumerate(intensity_profile):
            if intensity <= min_intensity + 10:
                dark_indices.append(i)

        if len(dark_indices) < 5:  # Minimum shadow width
            return None

        # Return the middle of the dark region
        shadow_center = start_x + (dark_indices[0] + dark_indices[-1]) // 2
        return shadow_center

    def find_geometric_split(self, image: np.ndarray) -> int:
        """
        Find split line using edge detection and geometric analysis.

        Args:
            image: Input image as numpy array

        Returns:
            X coordinate of split line
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Look for vertical lines in the center region
        center_region = width // 3
        start_x = width // 2 - center_region // 2
        end_x = width // 2 + center_region // 2

        # Calculate vertical line strength
        line_scores = []
        for x in range(start_x, end_x):
            column = edges[:, x]
            line_strength = np.sum(column > 0)
            line_scores.append(line_strength)

        # Find the strongest vertical line
        if line_scores:
            max_score_idx = np.argmax(line_scores)
            return start_x + max_score_idx

        # Fallback to center
        return width // 2

    def split_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dual page image into two single pages.

        Args:
            image: Input dual-page image

        Returns:
            Tuple of (left_page, right_page)
        """
        height, width = image.shape[:2]

        # Try to find shadow split first
        split_x = self.find_shadow_split(image)

        if split_x is None:
            logger.debug("No shadow detected, using geometric split")
            split_x = self.find_geometric_split(image)
        else:
            logger.debug(f"Shadow split found at x={split_x}")

        # Split the image with small overlap to avoid cutting text
        overlap = 10
        left_page = image[:, :split_x + overlap]
        right_page = image[:, split_x - overlap:]

        return left_page, right_page

    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to input PDF

        Returns:
            List of page images as numpy arrays
        """
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # Render page to image
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # Scale for DPI
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            images.append(img)
            logger.info(f"Extracted page {page_num + 1}/{doc.page_count}")

        doc.close()
        return images

    def images_to_pdf(self, images: List[np.ndarray], output_path: str):
        """
        Convert list of images to PDF.

        Args:
            images: List of page images
            output_path: Output PDF path
        """
        if not images:
            raise ValueError("No images to convert")

        # Create PDF
        c = canvas.Canvas(output_path, pagesize=A4)

        for i, img in enumerate(images):
            logger.info(f"Adding page {i + 1}/{len(images)} to PDF")

            # Convert OpenCV image to PIL
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                pil_img = Image.fromarray(img)

            # Get page dimensions
            img_width, img_height = pil_img.size
            page_width, page_height = A4

            # Calculate scaling to fit page while maintaining aspect ratio
            width_ratio = page_width / img_width
            height_ratio = page_height / img_height
            scale = min(width_ratio, height_ratio)

            # Calculate centered position
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            x = (page_width - scaled_width) / 2
            y = (page_height - scaled_height) / 2

            # Add image to PDF
            c.drawImage(ImageReader(pil_img), x, y, width=scaled_width, height=scaled_height)
            c.showPage()

        c.save()
        logger.info(f"PDF saved: {output_path}")

    def process_file(self, input_path: str, output_path: str, enhance: bool = True) -> str:
        """
        Process a single PDF file.

        Args:
            input_path: Path to input PDF
            output_path: Path to output PDF
            enhance: Whether to enhance text quality

        Returns:
            Output file path
        """
        logger.info(f"Processing: {input_path}")

        # Handle different file types
        input_ext = Path(input_path).suffix.lower()

        if input_ext == '.pdf':
            # Extract images from PDF
            page_images = self.pdf_to_images(input_path)
        else:
            # Single image file
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            page_images = [img]

        # Process each page
        processed_pages = []

        for page_num, page_img in enumerate(page_images):
            logger.info(f"Processing page {page_num + 1}/{len(page_images)}")

            # Split the page
            left_page, right_page = self.split_page(page_img)

            # Enhance text if requested
            if enhance:
                left_page = self.enhance_text(left_page)
                right_page = self.enhance_text(right_page)

            # Add pages in reading order (left first, then right)
            processed_pages.extend([left_page, right_page])

        # Convert to PDF
        self.images_to_pdf(processed_pages, output_path)

        logger.info(f"Successfully processed {len(page_images)} dual pages into {len(processed_pages)} single pages")
        return output_path

    def process_directory(self, input_dir: str, output_dir: str, enhance: bool = True) -> List[str]:
        """
        Process all files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            enhance: Whether to enhance text quality

        Returns:
            List of output file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Supported file extensions
        extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

        # Find all supported files
        input_path = Path(input_dir)
        files = [f for f in input_path.iterdir()
                 if f.suffix.lower() in extensions]

        # Sort files naturally
        files.sort(key=lambda x: x.name)

        outputs = []
        for file_path in files:
            try:
                # Generate output filename
                output_name = f"{file_path.stem}_split.pdf"
                output_path = os.path.join(output_dir, output_name)

                # Process file
                result = self.process_file(str(file_path), output_path, enhance)
                outputs.append(result)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Split dual-page PDFs into single-page PDFs",
        epilog="""
USAGE EXAMPLES:
  # Just split pages (recommended default):
  python script.py "book.pdf" --no-enhance

  # Split all PDFs in current directory:
  python script.py --no-enhance

  # Split with text enhancement (for poor scans only):
  python script.py "book.pdf"

  # Process multiple files:
  python script.py *.pdf --no-enhance
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", help="Input PDF files (if none specified, processes current directory)")
    parser.add_argument("--output", "-o", default="scan_fix_output",
                        help="Output directory (default: scan_fix_output)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip text enhancement (RECOMMENDED for normal scans)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PDF rendering (default: 300)")
    parser.add_argument("--shadow-threshold", type=int, default=50,
                        help="Shadow detection threshold (0-255)")
    parser.add_argument("--min-shadow-height", type=float, default=0.3,
                        help="Minimum shadow height ratio (0-1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize splitter
    splitter = PDFPageSplitter(
        shadow_threshold=args.shadow_threshold,
        min_shadow_height=args.min_shadow_height,
        dpi=args.dpi
    )

    enhance = not args.no_enhance

    # Show what we're doing
    if enhance:
        print("🔧 Running with text enhancement (for poor quality scans)")
        print("💡 For normal scans, use --no-enhance flag for better results")
    else:
        print("📄 Running page split only (no text enhancement)")

    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # If no files specified, process current directory
        if not args.files:
            print("No files specified, processing current directory...")
            current_dir = "."
            outputs = splitter.process_directory(current_dir, args.output, enhance)
            print(f"Successfully processed {len(outputs)} files in {args.output}/")
            return 0

        # Process specified files
        processed_count = 0
        for file_path in args.files:
            try:
                if os.path.isfile(file_path):
                    # Process single file
                    input_stem = Path(file_path).stem
                    output_path = os.path.join(args.output, f"{input_stem}_split.pdf")

                    splitter.process_file(file_path, output_path, enhance)
                    print(f"✓ Processed: {file_path} → {output_path}")
                    processed_count += 1

                elif os.path.isdir(file_path):
                    # Process directory
                    outputs = splitter.process_directory(file_path, args.output, enhance)
                    print(f"✓ Processed directory: {file_path} ({len(outputs)} files)")
                    processed_count += len(outputs)

                else:
                    print(f"⚠ Skipping: {file_path} (not found)")

            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")

        print(f"\nDone! Processed {processed_count} files to {args.output}/")

        if enhance:
            print("\n💡 TIP: If output looks over-processed, try using --no-enhance flag")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
