"""
PDF Dual Page Splitter & Text Enhancer
Converts dual-page PDFs to single-page PDFs with optional text enhancement for OCR.
Handles both evenly split pages and angled scans with shadows.

USAGE EXAMPLES:
    # Just split pages (no enhancement) - DEFAULT
    python script.py "book.pdf"

    # Split all PDFs in the current directory (no enhancement)
    python script.py

    # Split with text enhancement (for poor quality scans only)
    python script.py "book.pdf" --enhance

    # Process multiple files without enhancement
    python script.py *.pdf

    # High quality scan processing (no enhancement)
    python script.py "book.pdf" --dpi 600

OUTPUT:
    - Creates scan_fix_output/ directory
    - Generates filename_split.pdf for each input
    - Left page first, then right page (proper reading order)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
# from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Configure logging (default to WARNING; controlled by --verbosity)
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFPageSplitter:
    def __init__(
        self, shadow_threshold: int = 50, min_shadow_height: float = 0.3, dpi: int = 300
    ):
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
            image: Input image as a numpy array

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
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Ensure we don't go beyond original bounds
        result = np.clip(sharpened, 0, 255).astype(np.uint8)

        return result

    def find_shadow_split(self, image: np.ndarray) -> Optional[int]:
        """
        Find the center split line by detecting page shadows.

        Args:
            image: Input image as a numpy array

        Returns:
            X coordinate of split line, or None if not found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        center_region = width // 4  # Search in the center 50% of image
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
        Find the split line using edge detection and geometric analysis.

        Args:
            image: Input image as a numpy array

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
            # np.argmax can return numpy integer types; cast to built-in int for typing/tooling
            max_score_idx = int(np.argmax(line_scores))
            return start_x + max_score_idx

        # Fallback to center
        return width // 2

    def split_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Split a dual-page image into two single pages.

        Args:
            image: Input dual-page image

        Returns:
            Tuple of (left_page, right_page, split_x)
        """
        height, width = image.shape[:2]

        # Try to find the shadow split first
        split_x = self.find_shadow_split(image)

        if split_x is None:
            logger.debug("No shadow detected, using geometric split")
            split_x = self.find_geometric_split(image)
        else:
            logger.debug(f"Shadow split found at x={split_x}")

        # Split the image with a small overlap to avoid cutting text
        overlap = 10
        left_page = image[:, : split_x + overlap]
        right_page = image[:, split_x - overlap :]

        return left_page, right_page, split_x

    def _append_imagemagick_command(self, output_dir: str, command: str):
        """Append a command line to the scan-split-imagemagick-command.txt file in output_dir."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            cmd_file = os.path.join(output_dir, "scan-split-imagemagick-command.txt")
            with open(cmd_file, "a", encoding="utf-8") as fh:
                fh.write(command.rstrip() + "\n")
            logger.debug(f"Appended ImageMagick command to: {cmd_file}")
        except Exception as e:
            logger.error(f"Failed to append ImageMagick command to {output_dir}: {e}")

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

            # Convert to a numpy array
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            images.append(img)
            # TODO convert to tqdm
            logger.info(f"Extracted page {page_num + 1}/{doc.page_count}")

        doc.close()
        return images

    def images_to_pdf(self, images: List[np.ndarray], output_path: str):
        """
        Convert the list of images to PDF.

        Args:
            images: List of page images
            output_path: Output PDF path
        """
        if not images:
            raise ValueError("No images to convert")

        # Create PDF
        c = canvas.Canvas(output_path, pagesize=A4)

        # TODO convert to tqdm
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

            # Calculate scaling to fit the page while maintaining the aspect ratio
            width_ratio = page_width / img_width
            height_ratio = page_height / img_height
            scale = min(width_ratio, height_ratio)

            # Calculate centered position
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            x = (page_width - scaled_width) / 2
            y = (page_height - scaled_height) / 2

            # Add image to PDF
            c.drawImage(
                ImageReader(pil_img), x, y, width=scaled_width, height=scaled_height
            )
            c.showPage()

        c.save()
        logger.info(f"PDF saved: {output_path}")

    def process_file_pdf(
        self,
        input_path: str,
        output_path: str,
        enhance: bool = False,
        imagemagick_commands: bool = True,
    ) -> str:
        """
        Process a single PDF file and produce a split PDF. Behavior unchanged for PDFs.
        """
        logger.info(f"Processing PDF: {input_path}")

        # Extract images from PDF
        page_images = self.pdf_to_images(input_path)

        # Process each page
        processed_pages = []
        # TODO convert to tqdm
        for page_num, page_img in enumerate(page_images):
            logger.debug(f"Processing page {page_num + 1}/{len(page_images)}")

            # Split the page
            left_page, right_page, split_x = self.split_page(page_img)
            logger.debug(f"Split page {page_num + 1} at x={split_x}")

            # Enhance text if requested
            if enhance:
                left_page = self.enhance_text(left_page)
                right_page = self.enhance_text(right_page)

            # Add pages in reading order (left first, then right)
            processed_pages.extend([left_page, right_page])

            # Optionally write ImageMagick commands for this page to the output directory
            try:
                out_dir = os.path.dirname(output_path) or "."
                # Build sensible PNG output names for manual splitting by user
                stem = Path(input_path).stem
                pad = len(str(len(page_images)))
                number_str = str(page_num + 1).zfill(pad)
                left_out = os.path.join(
                    out_dir, f"{stem}_page{number_str}_split_{number_str}_1.png"
                )
                right_out = os.path.join(
                    out_dir, f"{stem}_page{number_str}_split_{number_str}_2.png"
                )

                # Use page specifier for PDFs so ImageMagick addresses the correct page
                input_spec = f"{input_path}[{page_num}]"

                # Build a bash one-liner that declares split_x and crops using ImageMagick `convert`.
                # Use .format() and double braces for shell vars so Python's formatter leaves ${height} untouched.
                cmd = (
                    "split_x={split_x}; width=$(identify -format \"%w\" '{input_spec}'); "
                    "height=$(identify -format \"%h\" '{input_spec}'); "
                    "convert '{input_spec}' -crop \"${{split_x}}x${{height}}+0+0\" +repage '{left_out}'; "
                    "convert '{input_spec}' -crop \"$((width - split_x))x${{height}}+${{split_x}}+0\" +repage '{right_out}'"
                ).format(
                    input_spec=input_spec,
                    split_x=split_x,
                    left_out=left_out,
                    right_out=right_out,
                )
                # Only append if caller enabled this feature (we'll check caller side)
                # We'll append unconditionally here; caller must call this block when enabled.
                # To keep control, we expect process_directory callers to call this block conditionally.
                # Append command to out_dir file
                # Note: we don't execute it here, just save it for user experimentation
                # We'll only append if output directory exists (it should)
                # Append handled by caller in process_directory by passing flag; here we only prepare cmd
                if imagemagick_commands:
                    if not hasattr(self, "_imagemagick_queue"):
                        self._imagemagick_queue = []
                    self._imagemagick_queue.append((out_dir, cmd))
            except Exception:
                logger.debug(
                    "Failed to prepare ImageMagick command for PDF page", exc_info=True
                )

        # Convert to PDF
        self.images_to_pdf(processed_pages, output_path)

        logger.info(
            f"Successfully processed {len(page_images)} dual pages into {len(processed_pages)} single pages"
        )
        return output_path

    def process_file_image(
        self,
        input_path: str,
        output_dir: str,
        index: int,
        total: int,
        enhance: bool = False,
        convert_image_to: Optional[str] = None,
        imagemagick_commands: bool = True,
    ) -> List[str]:
        """
        Process a single image file and output two split image files.
        """
        logger.debug(f"Processing image: {input_path}")

        # Load image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")

        # Split into left and right pages
        left_page, right_page, split_x = self.split_page(img)
        logger.debug(f"Split image {input_path} at x={split_x}")

        # Optional enhancement
        if enhance:
            left_page = self.enhance_text(left_page)
            right_page = self.enhance_text(right_page)

        # Build output filenames
        pad = len(str(total))
        number_str = str(index).zfill(pad)
        in_path = Path(input_path)
        stem = in_path.stem
        ext = in_path.suffix
        out1 = os.path.join(output_dir, f"{stem}_split_{number_str}_1{ext}")
        out2 = os.path.join(output_dir, f"{stem}_split_{number_str}_2{ext}")

        os.makedirs(output_dir, exist_ok=True)
        ok1 = cv2.imwrite(out1, left_page)
        ok2 = cv2.imwrite(out2, right_page)
        if not ok1 or not ok2:
            raise IOError(f"Failed to write output images for {input_path}")

        logger.debug(f"Wrote: {out1} and {out2}")

        # Optionally append ImageMagick command for manual splitting
        # The process_directory caller will set self._imagemagick_queue when enabled; append now if present
        try:
            if imagemagick_commands:
                if not hasattr(self, "_imagemagick_queue"):
                    self._imagemagick_queue = []
                cmd = (
                    "split_x={split_x}; width=$(identify -format \"%w\" '{input_path}'); "
                    "height=$(identify -format \"%h\" '{input_path}'); "
                    "convert '{input_path}' -crop \"${{split_x}}x${{height}}+0+0\" +repage '{out1}'; "
                    "convert '{input_path}' -crop \"$((width - split_x))x${{height}}+${{split_x}}+0\" +repage '{out2}'"
                ).format(input_path=input_path, split_x=split_x, out1=out1, out2=out2)
                self._imagemagick_queue.append((output_dir, cmd))
        except Exception:
            logger.debug(
                "Failed to prepare ImageMagick command for image", exc_info=True
            )

        # Convert output images to the desired format if specified
        if convert_image_to:
            converted_files = []
            for output_file in [out1, out2]:
                # Determine the new file name
                new_file = str(Path(output_file).with_suffix(f".{convert_image_to}"))

                # Read the image
                image = cv2.imread(output_file)
                if image is None:
                    raise ValueError(
                        f"Could not load image for conversion: {output_file}"
                    )

                # Convert and save the image in the new format
                if convert_image_to == "jpg" or convert_image_to == "jpeg":
                    # For JPEG, we need to specify the quality
                    cv2.imwrite(new_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                elif convert_image_to == "png":
                    # For PNG, we can specify the compression level (0-9); 0 is no compression
                    cv2.imwrite(new_file, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                else:
                    # For other formats, save with default parameters
                    cv2.imwrite(new_file, image)

                logger.debug(f"Converted and saved: {new_file}")
                converted_files.append(new_file)

            return converted_files

        return [out1, out2]

    def process_file(
        self,
        input_path: str,
        output_path: str,
        enhance: bool = False,
        convert_image_to: Optional[str] = None,
    ) -> str:
        """
        Backward-compatible wrapper:
        - If input is PDF, writes a PDF to output_path.
        - If input is an image, writes two images to the directory of output_path and returns the first image path.
        """
        logger.info(f"Processing: {input_path}")
        input_ext = Path(input_path).suffix.lower()
        if input_ext == ".pdf":
            return self.process_file_pdf(input_path, output_path, enhance)
        else:
            output_dir = os.path.dirname(output_path) or "."
            outputs = self.process_file_image(
                input_path,
                output_dir,
                index=1,
                total=1,
                enhance=enhance,
                convert_image_to=convert_image_to,
            )
            return outputs[0]

    def flush_imagemagick_queue(self):
        """Write any queued ImageMagick commands to their files and clear the queue."""
        try:
            if hasattr(self, "_imagemagick_queue") and self._imagemagick_queue:
                for out_dir, cmd in self._imagemagick_queue:
                    try:
                        self._append_imagemagick_command(out_dir, cmd)
                    except Exception:
                        logger.debug(
                            "Failed to write one ImageMagick command", exc_info=True
                        )
                # Clear the queue
                try:
                    del self._imagemagick_queue
                except Exception:
                    pass
        except Exception:
            logger.debug("Error flushing ImageMagick queue", exc_info=True)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        enhance: bool = False,
        convert_image_to: Optional[str] = None,
    ) -> List[str]:
        """
        Process all supported files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            enhance: Whether to enhance text quality
            convert_image_to: Convert output images to this format (e.g., jpg, png). Default: keep the original format.

        Returns:
            List of output file paths (for images, returns both split files)
        """
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Supported file extensions
        extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".jp2"}

        # Find all supported files
        input_path = Path(input_dir)
        files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        # Sort files naturally
        files.sort(key=lambda x: x.name)

        total = len(files)
        outputs: List[str] = []
        # If enabled, prepare a queue to collect commands and flush them to file at the end
        # We attach it to self so inner methods can append when they create a split
        imagemagick_enabled = getattr(self, "_imagemagick_enabled_temp", None)
        # We'll set this based on caller's parameter; default to True for backward compatibility
        # Note: process_directory now expects callers to have set attribute before calling
        if getattr(self, "_imagemagick_enabled", None) is None:
            # default True unless caller sets _imagemagick_enabled
            self._imagemagick_enabled = True
        if self._imagemagick_enabled:
            self._imagemagick_queue = []

        # Prepare to process files
        # If there is more than one file and tqdm is available, show a progress bar
        if total > 1 and tqdm is not None and callable(tqdm):
            # `tqdm` returns an iterable; static type checkers may complain about arg types
            file_iter = enumerate(tqdm(files, desc="Processing files", unit="file"), start=1)  # type: ignore[arg-type]
        else:
            if total > 1 and tqdm is None:
                logger.debug("tqdm not installed; running without progress bar")
            file_iter = enumerate(files, start=1)

        for idx, file_path in file_iter:
            try:
                input_ext = file_path.suffix.lower()
                if input_ext == ".pdf":
                    # Generate output filename for PDF (unchanged behavior)
                    output_name = f"{file_path.stem}_split.pdf"
                    output_path = os.path.join(output_dir, output_name)
                    result = self.process_file_pdf(
                        str(file_path),
                        output_path,
                        enhance,
                        imagemagick_commands=getattr(
                            self, "_imagemagick_enabled", True
                        ),
                    )
                    outputs.append(result)
                else:
                    # Image: write split images (no PDF)
                    image_outputs = self.process_file_image(
                        str(file_path),
                        output_dir,
                        index=idx,
                        total=total,
                        enhance=enhance,
                        convert_image_to=convert_image_to,
                        imagemagick_commands=getattr(
                            self, "_imagemagick_enabled", True
                        ),
                    )
                    outputs.extend(image_outputs)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Flush any collected ImageMagick commands to the output file
        try:
            if getattr(self, "_imagemagick_enabled", False) and hasattr(
                self, "_imagemagick_queue"
            ):
                for out_dir, cmd in self._imagemagick_queue:
                    try:
                        self._append_imagemagick_command(out_dir, cmd)
                    except Exception:
                        logger.debug(
                            "Failed to write one ImageMagick command", exc_info=True
                        )
                # cleanup temporary queue
                delattr = False
                try:
                    del self._imagemagick_queue
                except Exception:
                    pass
        except Exception:
            logger.debug("Failed to flush ImageMagick commands", exc_info=True)

        return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split dual-page scans into single pages (PDFs unchanged; images saved as split images)",
        epilog="""
USAGE EXAMPLES:
  # Just split pages (no enhancement - DEFAULT):
  python script.py "book.pdf"

  # Split all supported files in current directory (no enhancement):
  python script.py

  # Split with text enhancement (for poor scans only):
  python script.py "book.pdf" --enhance

  # Process multiple files:
  python script.py *.pdf
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input files or directories (if none specified, processes current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scan-split-output",
        help="Output directory (default: scan-split-output)",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enable gentle text enhancement (OFF by default)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for PDF rendering (default: 300)"
    )
    parser.add_argument(
        "--shadow-threshold",
        type=int,
        default=50,
        help="Shadow detection threshold (0-255)",
    )
    parser.add_argument(
        "--min-shadow-height",
        type=float,
        default=0.3,
        help="Minimum shadow height ratio (0-1)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str,
        choices=["debug", "info", "warning", "warn", "error", "critical"],
        default="warning",
        help="Logging verbosity (default: warning)",
    )
    parser.add_argument(
        "--convert-image-to",
        "-c",
        type=str,
        default=None,
        help="Convert output images to this format (e.g., jpg, png). Default: keep original format.",
    )
    # Enable imagemagick commands by default; allow disabling with --no-imagemagick-commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--imagemagick-commands",
        dest="imagemagick_commands",
        action="store_true",
        help="Enable writing ImageMagick commands to files (scan-split-imagemagick-command.txt)",
    )
    group.add_argument(
        "--no-imagemagick-commands",
        dest="imagemagick_commands",
        action="store_false",
        help="Disable writing ImageMagick commands to files",
    )
    parser.set_defaults(imagemagick_commands=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Set logging level: --debug overrides --verbosity
    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Normalize 'warn' to 'warning'
        level_name = getattr(args, "verbosity", "warning")
        if isinstance(level_name, str) and level_name.lower() == "warn":
            level_name = "warning"
        try:
            level = getattr(logging, str(level_name).upper())
            if isinstance(level, int):
                logging.getLogger().setLevel(level)
            else:
                logging.getLogger().setLevel(logging.WARNING)
        except Exception:
            logging.getLogger().setLevel(logging.WARNING)

    # Initialize splitter
    splitter = PDFPageSplitter(
        shadow_threshold=args.shadow_threshold,
        min_shadow_height=args.min_shadow_height,
        dpi=args.dpi,
    )

    # Enable or disable ImageMagick command writing (default: enabled)
    splitter._imagemagick_enabled = args.imagemagick_commands

    enhance = args.enhance
    convert_image_to = args.convert_image_to

    # Show what we're doing
    if enhance:
        print("🔧 Running with text enhancement (for poor quality scans)")
        print("💡 For normal scans, omit --enhance for best results")
    else:
        print("📄 Running page split only (no text enhancement)")

    if convert_image_to:
        print(f"🖼️ Output images will be converted to: {convert_image_to.upper()}")

    try:
        # Create the output directory
        os.makedirs(args.output, exist_ok=True)

        # If no files specified, process the current directory
        if not args.files:
            print("No files specified, processing current directory...")
            outputs = splitter.process_directory(
                ".", args.output, enhance, convert_image_to=convert_image_to
            )
            print(f"Successfully processed {len(outputs)} files in {args.output}/")
            return 0

        # Prepare count for files (exclude directories) for image numbering
        file_args = [p for p in args.files if os.path.isfile(p)]
        total_files = len(file_args)
        file_index = 0

        # Process specified files and directories
        processed_count = 0
        for item in args.files:
            try:
                if os.path.isfile(item):
                    file_index += 1
                    input_ext = Path(item).suffix.lower()
                    if input_ext == ".pdf":
                        # Process PDF (unchanged behavior)
                        input_stem = Path(item).stem
                        output_path = os.path.join(
                            args.output, f"{input_stem}_split.pdf"
                        )
                        splitter.process_file_pdf(
                            item,
                            output_path,
                            enhance,
                            imagemagick_commands=splitter._imagemagick_enabled,
                        )
                        # Flush any queued commands for this single file
                        if splitter._imagemagick_enabled:
                            splitter.flush_imagemagick_queue()
                        print(f"✓ Processed: {item} → {output_path}")
                        processed_count += 1
                    else:
                        # Process image: write split images
                        outputs = splitter.process_file_image(
                            item,
                            args.output,
                            index=file_index,
                            total=total_files,
                            enhance=enhance,
                            convert_image_to=convert_image_to,
                            imagemagick_commands=splitter._imagemagick_enabled,
                        )
                        # Flush any queued commands for this single file
                        if splitter._imagemagick_enabled:
                            splitter.flush_imagemagick_queue()
                        for outp in outputs:
                            print(f"✓ Wrote: {outp}")
                        processed_count += len(outputs)
                elif os.path.isdir(item):
                    # Process directory
                    outputs = splitter.process_directory(
                        item, args.output, enhance, convert_image_to=convert_image_to
                    )
                    print(f"✓ Processed directory: {item} ({len(outputs)} files)")
                    processed_count += len(outputs)
                else:
                    print(f"⚠ Skipping: {item} (not found)")
            except Exception as e:
                print(f"✗ Error processing {item}: {e}")

        print(f"\nDone! Processed {processed_count} files to {args.output}/")

        if enhance:
            print("\n💡 TIP: If output looks over-processed, run without --enhance")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
