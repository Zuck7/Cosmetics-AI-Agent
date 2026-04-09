from pathlib import Path
import os
import re
import importlib


def _normalize_text(text):
    if not text:
        return ""
    # Keep line breaks but collapse noisy spacing from PDF extraction.
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_pypdf2(pdf_path, password=None):
    import PyPDF2

    extracted = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f, strict=False)

        if reader.is_encrypted:
            # Try a provided password first, then blank password for owner-locked PDFs.
            if password:
                try:
                    reader.decrypt(password)
                except Exception:
                    pass
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            if reader.is_encrypted:
                raise ValueError("File is encrypted and could not be decrypted")

        for page in reader.pages:
            extracted.append(page.extract_text() or "")

    return _normalize_text("\n".join(extracted))


def _extract_with_pypdf(pdf_path, password=None):
    try:
        pypdf = importlib.import_module("pypdf")
        PdfReader = pypdf.PdfReader
    except Exception:
        return ""

    extracted = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f, strict=False)

        if reader.is_encrypted:
            if password:
                try:
                    reader.decrypt(password)
                except Exception:
                    pass
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            if reader.is_encrypted:
                return ""

        for page in reader.pages:
            extracted.append(page.extract_text() or "")

    return _normalize_text("\n".join(extracted))


def _extract_with_pymupdf(pdf_path):
    try:
        fitz = importlib.import_module("fitz")
        if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "mupdf_display_errors"):
            fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        return ""

    extracted = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            extracted.append(page.get_text("text") or "")
    return _normalize_text("\n".join(extracted))


def _extract_with_ocr(pdf_path):
    """
    Optional OCR fallback for scanned/image-only PDFs.
    Requires PyMuPDF + pytesseract + system tesseract binary.
    """
    try:
        fitz = importlib.import_module("fitz")
        if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "mupdf_display_errors"):
            fitz.TOOLS.mupdf_display_errors(False)
        pytesseract = importlib.import_module("pytesseract")
        from PIL import Image
    except Exception:
        return ""

    extracted = []
    max_pages = int(os.getenv("OCR_MAX_PAGES", "30"))
    zoom = float(os.getenv("OCR_ZOOM", "2.0"))
    matrix = fitz.Matrix(zoom, zoom)

    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                pix = page.get_pixmap(matrix=matrix)
                mode = "RGB" if pix.alpha == 0 else "RGBA"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                extracted.append(pytesseract.image_to_string(image) or "")
    except Exception:
        return ""

    return _normalize_text("\n".join(extracted))

def load_pdfs(folder_path):
    """Load PDFs from folder. Falls back to mock cosmetics data if no PDFs found."""
    try:
        import PyPDF2
    except ImportError:
        print("PyPDF2 not installed. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()

    folder = Path(folder_path)
    documents = []

    # Try to load PDFs from the folder
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {folder_path}. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()

    for pdf_path in pdf_files:
        text = ""
        try:
            password = os.getenv("PDF_PASSWORD", "")

            # Primary extraction via PyPDF2.
            text = _extract_with_pypdf2(pdf_path, password=password)

            # Secondary extraction via pypdf for PDFs that PyPDF2 cannot parse well.
            if len(text.split()) < 20:
                pypdf_text = _extract_with_pypdf(pdf_path, password=password)
                if len(pypdf_text.split()) > len(text.split()):
                    text = pypdf_text

            # Fallback extraction for PDFs where PyPDF2 returns near-empty content.
            if len(text.split()) < 20:
                alt_text = _extract_with_pymupdf(pdf_path)
                if len(alt_text.split()) > len(text.split()):
                    text = alt_text

            # OCR fallback for scanned/image-only files.
            if len(text.split()) == 0:
                ocr_text = _extract_with_ocr(pdf_path)
                if len(ocr_text.split()) > 0:
                    text = ocr_text

            word_count = len(text.split())
            if word_count == 0:
                print(
                    f"Warning: {pdf_path.name} produced no extractable text. "
                    "Likely scanned/image-only; OCR is required."
                )

            documents.append({
                "file": pdf_path.name,
                "text": text,
                "word_count": word_count,
            })
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}. Skipping.")

    if not documents:
        print(f"Could not load any PDFs. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()

    return documents


def _get_mock_cosmetics_data():
    """Provide mock cosmetics knowledge base for demo/testing."""
    return [
        {
            "file": "mock_cosmetics_ingredients.txt",
            "text": """
ANTI-AGING INGREDIENTS

Retinol: A form of Vitamin A used in skincare to reduce fine lines and wrinkles. 
It works by promoting cell turnover and stimulating collagen production. 
Retinol is typically used at night as it can increase sun sensitivity.

Hyaluronic Acid: A humectant that holds up to 1000 times its weight in water.
It helps plump the skin and reduce the appearance of fine lines and wrinkles.
Hyaluronic acid is suitable for all skin types, including sensitive skin.

Vitamin C: An antioxidant that protects against free radical damage and brightens the skin.
It also promotes collagen synthesis and can reduce hyperpigmentation.
Vitamin C serums are most effective when used in the morning before sunscreen.

Peptides: Short chains of amino acids that signal the skin to produce more collagen.
Peptides can help improve skin firmness and elasticity over time.
They work synergistically with other anti-aging ingredients.

Niacinamide: Also known as Vitamin B3, niacinamide reduces pore size and sebum production.
It also has anti-inflammatory properties and can strengthen the skin barrier.
Niacinamide is well-tolerated by most skin types.

NATURAL MOISTURIZING INGREDIENTS

Hyaluronic Acid: A humectant that retains moisture and plumps the skin.

Glycerin: A powerful humectant that draws moisture into the skin and maintains hydration.

Ceramides: Lipids that form the skin barrier and prevent transepidermal water loss (TEWL).

Plant Oils: Jojoba oil, argan oil, and rosehip seed oil provide emollient benefits.

SAFETY CONSIDERATIONS FOR COSMETICS

Sun Protection Factor (SPF): Daily sunscreen with SPF 30 or higher protects against UVA and UVB rays.
Regular sunscreen use can prevent premature aging, dark spots, and skin cancer risk.

Allergic Reactions: Patch test new products on a small area before full application.
Common allergens include fragrances, essential oils, and preservatives.

Sensitivity to Actives: Introduce active ingredients slowly to allow skin adaptation.
Using multiple actives simultaneously can compromise the skin barrier.

Pregnancy Safety: Some ingredients like retinol and salicylic acid are contraindicated during pregnancy.
Consult healthcare providers before using new skincare products.

INGREDIENT LABELING AND REGULATIONS

FDA Regulations: Cosmetics in the US must list ingredients in descending order of concentration.
The FDA does not require pre-market approval for most cosmetic ingredients.

Allergen Disclosure: Products must disclose fragrance components and known allergens.

Cruelty-Free and Vegan Certifications: These certifications ensure products are not tested on animals.

Organic Claims: Products labeled organic must meet USDA or similar standards.
            """
        },
        {
            "file": "mock_cosmetics_formulations.txt",
            "text": """
POPULAR COSMETIC FORMULATIONS

Foundation: Suspensions of pigments and binders in oil or water phases.
Modern foundations include SPF, hydrating ingredients, and long-wear formulas.
Formulations are tailored for different skin types and coverage levels.

Serums: Lightweight, high-concentration formulations of active ingredients.
Serums penetrate the skin quickly and can target specific concerns like aging or acne.
Common serums include Vitamin C, Hyaluronic Acid, and Retinol serums.

Face Masks: Concentrated treatments applied for 10-20 minutes for intensive care.
Clay masks absorb excess oil; gel masks provide hydration; sheet masks deliver targeted ingredients.

Moisturizers: Emulsions that balance oil and water to hydrate and protect the skin.
Lightweight hydrating moisturizers suit oily skin; rich creams suit dry skin.

COSMETIC PRESERVATION

Preservatives: Essential to prevent bacterial and fungal growth in products.
Common preservatives include parabens, phenoxyethanol, and natural alternatives like rosehip extract.

Antioxidants: Ingredients like Vitamin E and green tea extract extend shelf life and provide skin benefits.

Packaging: Airless pumps and opaque containers protect products from oxidation and contamination.

TRENDS IN COSMETICS

Clean Beauty: Products free from harsh chemicals, artificial fragrances, and synthetic dyes.

Sustainable Packaging: Brands are moving toward recyclable, biodegradable, and refillable options.

Personalization: AI and skin analysis tools help consumers find products tailored to their needs.

K-Beauty and J-Beauty: Asian skincare trends emphasizing prevention and layering of lightweight products.
            """
        }
    ]
