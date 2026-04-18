import os
import time
import argparse
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver(headless=False):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=chrome_options)

def download_pdfs(driver, ticker, folder):
    """Extracts PDF links from the current page and downloads them."""
    pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
    print(f"Found {len(pdf_links)} documents on this page.")

    for i, link in enumerate(pdf_links):
        url = link.get_attribute("href")
        # Clean title for filename
        raw_text = link.text.split('(')[0].strip() # Remove '(200KB)' suffix
        title = "".join(filter(lambda x: x.isalnum() or x in " -_", raw_text))[:50]
        
        # Unique timestamp/index to prevent overwriting
        filename = f"{int(time.time())}_{i}_{title}.pdf"
        filepath = os.path.join(folder, filename)
        
        try:
            r = requests.get(url, timeout=15)
            with open(filepath, 'wb') as f:
                f.write(r.content)
            print(f"  [+] Saved: {filename}")
        except Exception as e:
            print(f"  [!] Failed to download {url}: {e}")

def scrape_hkex(ticker, headless=False):
    # Dynamic Path Setup: data/ETF/documentation/{symbol}/
    base_dir = os.path.join("data", "ETF", "documentation", ticker)
    os.makedirs(base_dir, exist_ok=True)

    driver = setup_driver(headless)
    wait = WebDriverWait(driver, 15)

    try:
        driver.get("https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en")

        # 1. Handle Cookie Banner
        try:
            cookie = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie.click()
        except:
            pass

        # 2. Input Ticker
        input_field = wait.until(EC.element_to_be_clickable((By.ID, "searchStockCode")))
        input_field.send_keys(ticker)

        # 3. Select Dropdown
        suggestion_css = "#autocomplete-list-0 .autocomplete-suggestion"
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, suggestion_css)))
        first_suggestion = driver.find_element(By.CSS_SELECTOR, suggestion_css)
        driver.execute_script("arguments[0].click();", first_suggestion)


        # 5. Search
        search_xpath = "//a[contains(@class, 'filter__btn-applyFilters-js') and contains(text(), 'SEARCH')]"
        search_btn = wait.until(EC.element_to_be_clickable((By.XPATH, search_xpath)))
        driver.execute_script("arguments[0].click();", search_btn)
        
        # 6. Pagination Loop
        while True:
            # Wait for results table
            # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-responsive tbody tr")))
            
            # Download files on current page
            download_pdfs(driver, ticker, base_dir)

            # Check for 'Next' button
            try:
                next_btn = driver.find_element(By.LINK_TEXT, "Next")
                if "disabled" in next_btn.get_attribute("class"):
                    break
                driver.execute_script("arguments[0].click();", next_btn)
                print("Moving to next page...")
                time.sleep(2) # Prevent rate limiting
            except:
                print("No more pages.")
                break

    except Exception as e:
        print(f"Error occurred: {e}")
        driver.save_screenshot(f"error_{ticker}.png")
    finally:
        driver.quit()
        print(f"Finished scraping {ticker}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HKEX ETF Document Scraper")
    parser.add_argument("symbol", help="The ETF ticker (e.g., 02800)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    # Standardize ticker to 5 digits (e.g., '2800' -> '02800')
    formatted_ticker = args.symbol.zfill(5)
    
    scrape_hkex(formatted_ticker, args.headless)