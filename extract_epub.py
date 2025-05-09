import json
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import urllib.parse # Still useful for potential future URL needs, but not for filename derivation now

# --- Configuration ---

# 1. Load your metadata
# --- OR --- Load from a JSON file ---
json_metadata_file = '/workspace/thviet/LLMs/Monolingual/QG27.73-viEduQALLMs/book_metadata.json'
try:
     with open(json_metadata_file, 'r', encoding='utf-8') as f:
         all_book_metadata = json.load(f)
except FileNotFoundError:
     print(f"Error: Metadata file not found at {json_metadata_file}")
     exit()
except json.JSONDecodeError:
     print(f"Error: Could not decode JSON from {json_metadata_file}")
     exit()


# 2. !!! IMPORTANT !!! Specify the path to the folder containing your EPUB files
#    Based on your image, it looks like 'nhasachmienphi_pdfs'
EPUB_FOLDER_PATH = "QG27.73-viEduQALLMs/nhasachmienphi_pdfs" # <--- *** SET THIS TO THE CORRECT FULL PATH if needed ***
                                        # e.g., "C:/Users/YourUser/Downloads/nhasachmienphi_pdfs"
                                        # or "/home/youruser/Downloads/nhasachmienphi_pdfs"

# 3. Optional: Where to save the extracted text (or None to just print)
OUTPUT_FOLDER_PATH = "QG27.73-viEduQALLMs/extracted_texts" # <--- *** SET THIS PATH or set to None ***
# Make sure this output folder exists, or the script will create it
if OUTPUT_FOLDER_PATH and not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)
    print(f"Created output directory: {OUTPUT_FOLDER_PATH}")

# --- Helper Function to Extract Text (same as before) ---
def extract_text_from_epub(file_path):
    """Reads an EPUB file and extracts text content from HTML items."""
    try:
        book = epub.read_epub(file_path)
        full_text = []
        items_to_process = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)

        for item in items_to_process:
            content = item.get_content()
            soup = BeautifulSoup(content, 'lxml') # Use 'html.parser' if lxml not installed
            text = soup.get_text(separator=' ', strip=True)
            if text:
                full_text.append(text)

        return "\n\n".join(full_text)

    except FileNotFoundError:
        print(f"  Error (within function): EPUB file not found at {file_path}")
        return None
    except ebooklib.epub.EpubException as e:
        print(f"  Error: Could not read EPUB file {file_path}. It might be corrupt or DRM protected. Details: {e}")
        return None
    except Exception as e:
        print(f"  Error processing EPUB {file_path}: {e}")
        return None

# --- Main Processing Logic ---
processed_count = 0
skipped_criteria_count = 0
skipped_file_not_found_count = 0
skipped_missing_filename_field = 0
error_count = 0

print(f"Starting text extraction process.")
print(f"Looking for EPUB files in: {os.path.abspath(EPUB_FOLDER_PATH)}") # Show absolute path
if OUTPUT_FOLDER_PATH:
    print(f"Extracted text will be saved to: {os.path.abspath(OUTPUT_FOLDER_PATH)}")
else:
    print("Extracted text will be printed to console (snippets).")

for book_meta in all_book_metadata:
    title = book_meta.get("title", "Unknown Title")
    downloads = book_meta.get("downloads", {})
    pdf_link = downloads.get("pdf")
    epub_link = downloads.get("epub")
    # Get the filename field provided in the metadata
    filename_from_meta = book_meta.get("filename")

    print("-" * 20)
    print(f"Processing metadata for: '{title}'")

    # *** Core Condition: Check if PDF link is missing AND EPUB link exists ***
    if pdf_link is None and epub_link:
        print(f"  Condition met: No PDF link, has EPUB link.")

        # --- Determine the expected LOCAL EPUB filename ---
        if not filename_from_meta:
            print(f"  ERROR: Skipping '{title}' because the 'filename' field is missing in the metadata.")
            skipped_missing_filename_field += 1
            continue # Skip to the next book

        try:
            # Construct the expected EPUB filename by replacing the extension
            base_name, _ = os.path.splitext(filename_from_meta)
            expected_epub_filename = base_name + ".epub"
            print(f"  Derived expected local EPUB filename: '{expected_epub_filename}' (from metadata 'filename' field)")

        except Exception as e:
            print(f"  ERROR: Could not derive EPUB filename from metadata field '{filename_from_meta}' for '{title}': {e}. Skipping.")
            # This shouldn't typically happen with os.path.splitext, but good practice
            error_count += 1
            continue

        # --- Construct the full path to the local EPUB file ---
        epub_full_path = os.path.join(EPUB_FOLDER_PATH, expected_epub_filename)

        # --- Check if the file actually exists in your local folder ---
        if os.path.exists(epub_full_path):
            print(f"  Found local EPUB file: {epub_full_path}")

            # --- Extract text from the found EPUB file ---
            book_text = extract_text_from_epub(epub_full_path)

            if book_text is not None: # Check for None explicitly
                processed_count += 1
                print(f"  Successfully extracted text. Length: {len(book_text)} chars.")

                # --- Output the text ---
                if OUTPUT_FOLDER_PATH:
                    # Create a safe filename for the output text file (using title)
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
                    output_txt_filename = os.path.join(OUTPUT_FOLDER_PATH, f"{safe_title}.txt")
                    try:
                        with open(output_txt_filename, 'w', encoding='utf-8') as f_out:
                            f_out.write(book_text)
                        print(f"  Saved text to: {output_txt_filename}")
                    except Exception as e:
                        print(f"  ERROR: Could not save text file {output_txt_filename}: {e}")
                        error_count += 1
                else:
                    # Print a snippet if not saving to file
                    print("\n  --- Extracted Text Snippet ---")
                    print(book_text[:500] + ("..." if len(book_text) > 500 else ""))
                    print("  --- End Snippet ---\n")

            else:
                # Extraction function already printed the specific error
                error_count += 1
                print(f"  Failed to extract text from '{title}' (EPUB file: {expected_epub_filename}).")

        else:
            # The file expected based on the metadata was not found locally
            print(f"  ERROR: Local EPUB file NOT FOUND at expected path: {epub_full_path}")
            print(f"  (Check if '{expected_epub_filename}' exists in '{os.path.abspath(EPUB_FOLDER_PATH)}')")
            skipped_file_not_found_count += 1

    else:
        # Log why it was skipped
        if pdf_link is not None:
            print(f"  Skipping '{title}': Has a PDF link.")
        elif epub_link is None:
            print(f"  Skipping '{title}': Does not have an EPUB link.")
        else:
             print(f"  Skipping '{title}': Does not meet criteria.") # Should not happen if logic is correct
        skipped_criteria_count += 1

print("=" * 30)
print("\nProcessing Summary:")
print(f"  Books processed (text extracted): {processed_count}")
print(f"  Books skipped (had PDF or no EPUB link): {skipped_criteria_count}")
print(f"  Books skipped (metadata 'filename' field missing): {skipped_missing_filename_field}")
print(f"  Books skipped (local EPUB file not found): {skipped_file_not_found_count}")
print(f"  Errors during EPUB reading/saving text: {error_count}")
print("=" * 30)