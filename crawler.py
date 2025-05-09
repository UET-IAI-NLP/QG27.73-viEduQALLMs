import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re
from tqdm import tqdm
import concurrent.futures
import json

# Configuration
BASE_URL = "https://nhasachmienphi.com/"
PDF_BASE_URL = "https://file.nhasachmienphi.com/pdf/"
OUTPUT_FOLDER = "nhasachmienphi_pdfs"
METADATA_FILE = "book_metadata.json"
DELAY = 1  # seconds between requests
MAX_RETRIES = 3
MAX_WORKERS = 5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def sanitize_filename(title):
    """Convert book title to safe filename"""
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip()

def download_file(url, filename):
    """Download file with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception:
            time.sleep(DELAY * (attempt + 1))
    return False

def extract_book_info(book_url):
    """Extract detailed book info from individual book page"""
    try:
        response = requests.get(book_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = None
        title_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8 col-lg-8')
        if title_div:
            title = title_div.find('h1').get_text().strip() if title_div.find('h1') else None
        
        # Extract author
        author = None
        for div in soup.find_all('div', class_='mg-t-10'):
            if 'Tác giả:' in div.get_text():
                author = div.get_text().replace('Tác giả:', '').strip()
                break
        
        # Extract categories
        categories = []
        for div in soup.find_all('div', class_='mg-tb-10'):
            if 'Thể loại:' in div.get_text():
                categories = [a.get_text().strip() for a in div.find_all('a')]
                break
        
        # Extract download links
        download_links = {
            'pdf': None,
            'epub': None,
            'mobi': None
        }
        
        for link in soup.select('a.button'):
            if 'pdf' in link.get('class', []):
                download_links['pdf'] = link['href']
            elif 'epub' in link.get('class', []):
                download_links['epub'] = link['href']
            elif 'mobi' in link.get('class', []):
                download_links['mobi'] = link['href']
        
        return {
            'url': book_url,
            'title': title,
            'author': author,
            'categories': categories,
            'downloads': download_links,
            'filename': f"{sanitize_filename(title)}.pdf" if title else None
        }
    
    except Exception as e:
        print(f"Error extracting info from {book_url}: {str(e)}")
        return None

def get_book_links_from_page(url):
    """Extract book links from a single page with more robust detection"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        book_links = []
        
        # Method 1: Look for book items with images
        for item in soup.select('div.item_sach'):
            link = item.find('a', href=True)
            if link and (link['href'].endswith('.html') or '/sach-' in link['href']):
                book_links.append(urljoin(BASE_URL, link['href']))
        
        # Method 2: Look for book title links (fallback)
        if not book_links:
            for link in soup.select('a.title_sach, h4 a[href*="/"]'):
                if link['href'].endswith('.html') or '/sach-' in link['href']:
                    book_links.append(urljoin(BASE_URL, link['href']))
        
        # Method 3: Look for any links that look like book pages (last resort)
        if not book_links:
            for link in soup.select('a[href*="/"]'):
                href = link['href']
                if (href.endswith('.html') or '/sach-' in href) and not any(x in href for x in ['/category/', '/page/']):
                    book_links.append(urljoin(BASE_URL, href))
        
        return list(set(book_links))  # Remove duplicates
    
    except Exception as e:
        print(f"Error processing page {url}: {str(e)}")
        return []

def get_all_pages_in_category(category_url):
    """Get all paginated pages for a category by finding the 'next' link."""
    pages = [category_url]
    current_page_url = category_url
    processed_urls = {category_url} # Keep track of processed URLs to avoid loops

    while True:
        print(f"  Fetching pagination info from: {current_page_url}")
        try:
            response = requests.get(current_page_url, timeout=15) # Use GET, longer timeout
            response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the pagination navigation block
            pagenavi_div = soup.find('div', class_='wp-pagenavi')
            if not pagenavi_div:
                print(f"  No pagination div found on {current_page_url}. Assuming single page.")
                break # No pagination controls found

            # Find the 'next' page link more reliably
            # Method 1: Look for a link that is the direct next sibling of the 'current' page span
            current_span = pagenavi_div.find('span', class_='current')
            next_page_link = None
            if current_span:
                next_sibling = current_span.find_next_sibling()
                if next_sibling and next_sibling.name == 'a' and 'page' in next_sibling.get('href', ''):
                   next_page_link = next_sibling

            # Method 2: Fallback - look for the link with class 'nextpostslink' or rel='next'
            if not next_page_link:
                 next_page_link = pagenavi_div.find('a', class_='nextpostslink', href=True)
                 if not next_page_link:
                     next_page_link = pagenavi_div.find('a', rel='next', href=True) # Common pattern

            if next_page_link and next_page_link['href']:
                next_page_url = urljoin(current_page_url, next_page_link['href'])

                # Check if we've seen this URL before (avoid infinite loops on bad pagination)
                if next_page_url in processed_urls:
                    print(f"  Detected pagination loop at {next_page_url}. Stopping.")
                    break

                # Check if the next URL seems valid (e.g., still contains /page/)
                if '/page/' not in next_page_url and current_page_url != category_url :
                     print(f"  Next page link '{next_page_url}' doesn't look like a standard page. Stopping.")
                     break

                print(f"  Found next page: {next_page_url}")
                pages.append(next_page_url)
                processed_urls.add(next_page_url)
                current_page_url = next_page_url
                time.sleep(DELAY) # Add delay between checking pages
            else:
                print(f"  No 'next' page link found on {current_page_url}. Reached the end.")
                break # No 'next' link found

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching page {current_page_url} for pagination: {str(e)}")
            break # Stop if there's a request error
        except Exception as e:
             print(f"  Error parsing pagination on {current_page_url}: {str(e)}")
             break

    print(f"  Finished pagination check for {category_url}. Found {len(pages)} pages.")
    return pages
# --- End of Modified Function ---

def process_category(category_url):
    """Process all pages in a category with better error handling"""
    all_books_in_category = [] # Changed variable name for clarity
    print(f"Getting pages for category: {category_url}")
    category_pages = get_all_pages_in_category(category_url) # Call the modified function

    if not category_pages:
        print(f"Warning: No pages found or error during pagination for category {category_url}")
        return []

    print(f"Processing {len(category_pages)} pages for {category_url}")
    for page_url in tqdm(category_pages, desc=f"Pages in {category_url.split('/')[-2]}", leave=False): # Add tqdm here for page progress
        time.sleep(DELAY)
        try:
            book_links = get_book_links_from_page(page_url)
            if book_links:
                all_books_in_category.extend(book_links)
                # Removed the print statement here to avoid clutter, tqdm shows progress
                # print(f"Found {len(book_links)} books on page: {page_url}")
            else:
                print(f"No books found on page: {page_url} - checking page structure")
                # Debug: Save page content for analysis ONLY if no books found
                try:
                    debug_response = requests.get(page_url)
                    with open('debug_page.html', 'w', encoding='utf-8') as f:
                        f.write(debug_response.text)
                    print(f"Saved structure of {page_url} to debug_page.html")
                except Exception as debug_e:
                    print(f"Could not save debug page for {page_url}: {debug_e}")

        except Exception as e:
            print(f"Error processing page {page_url}: {str(e)}")

    return all_books_in_category
# --- End of Modified Function ---
def get_all_category_urls():
    """Get all category URLs from the main menu"""
    try:
        response = requests.get(BASE_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        return list(set(
            a['href'] for a in soup.select('li.menu-item a[href*="/category/"]')
            if '/category/' in a['href']
        ))
    except Exception as e:
        print(f"Error getting categories: {str(e)}")
        return []

def download_book_files(book_info):
    """Download all available formats for a book"""
    if not book_info or not book_info.get('downloads'):
        return None
    
    downloaded = {}
    for format, url in book_info['downloads'].items():
        if url:
            filename = f"{sanitize_filename(book_info['title'])}.{format}" if book_info['title'] else None
            if filename and download_file(url, filename):
                downloaded[format] = filename
    
    return downloaded if downloaded else None

def save_metadata(metadata):
    """Save book metadata to JSON file"""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def main():
    print("Starting nhasachmienphi.com crawler with proper pagination")
    
    # Step 1: Get all category URLs
    categories = get_all_category_urls()
    print(f"Found {len(categories)} categories")
    
    # Step 2: Process each category to get all book URLs
    all_book_urls = []
    for category in tqdm(categories, desc="Processing categories"):
        print(f"\nProcessing category: {category}")
        book_urls = process_category(category)
        all_book_urls.extend(book_urls)
        print(f"Total books found in category: {len(book_urls)}")
    
    # Remove duplicate URLs
    unique_book_urls = list(set(all_book_urls))
    print(f"\nFound {len(unique_book_urls)} unique book pages")
    
    # Step 3: Extract detailed info from each book page
    all_book_info = []
    for book_url in tqdm(unique_book_urls, desc="Extracting book info"):
        time.sleep(DELAY)
        book_info = extract_book_info(book_url)
        if book_info:
            all_book_info.append(book_info)
    
    # Save metadata
    save_metadata(all_book_info)
    print(f"Saved metadata to {METADATA_FILE}")
    
    # Step 4: Download files with concurrent threads
    downloaded_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_book_files, book) for book in all_book_info]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Downloading files"):
            if future.result():
                downloaded_count += 1
    
    print(f"\nDownload complete! Downloaded files for {downloaded_count}/{len(all_book_info)} books")
    print(f"Saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()