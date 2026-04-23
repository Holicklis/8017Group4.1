import os
import time
import requests
import logging
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("etp_prospectus.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_driver(headless=False):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=chrome_options)

def download_pdfs(driver, ticker, folder):
    """Extracts and saves PDF links from the current result page."""
    try:
        # Wait for the results table to appear
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "table")))
        pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
        logger.info(f"[{ticker}] Found {len(pdf_links)} document(s) on this page.")

        for i, link in enumerate(pdf_links):
            url = link.get_attribute("href")
            raw_text = link.text.split('(')[0].strip()
            title = "".join(filter(lambda x: x.isalnum() or x in " -_", raw_text))[:60]
            
            filename = f"{ticker}_{int(time.time())}_{i}_{title}.pdf"
            filepath = os.path.join(folder, filename)
            
            try:
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
                if r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
                    logger.info(f"  [+] Saved: {filename}")
            except Exception as e:
                logger.error(f"  [!] Failed {url}: {e}")
    except Exception:
        logger.info(f"[{ticker}] No document table found.")

def scrape_etp_prospectus(ticker, headless=False):
    # Standardize ticker (e.g., 2800)
    formatted_ticker = str(ticker).lstrip('0') # HKEX Quote URL usually takes '2800'
    base_dir = os.path.join("data", "ETF", "documentation", str(ticker).zfill(5), "pdf")
    os.makedirs(base_dir, exist_ok=True)

    driver = setup_driver(headless)
    wait = WebDriverWait(driver, 15)

    try:
        # 1. Go to Quote Page
        url = f"https://www.hkex.com.hk/Market-Data/Securities-Prices/Exchange-Traded-Products/Exchange-Traded-Products-Quote?sym={formatted_ticker}&sc_lang=en"
        logger.info(f"Navigating to Quote Page: {url}")
        driver.get(url)

        # 2. Click "Prospectus" link (using the div class you provided)
        # This usually triggers a JavaScript redirect to HKEXNews
        logger.info("Triggering Prospectus redirect...")
        prospectus_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.sublink.prospectus h2 a")))
        driver.execute_script("arguments[0].click();", prospectus_link)
        
        # 3. Handle the switch to the new window/tab if it opens
        time.sleep(3) # Give it time to load the new page
        if len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])

        # 4. Search on HKEXNews (Pre-filtered)
        # logger.info("Executing Search on pre-filtered page...")
        # Since it's pre-filtered, we just need to hit the Search button
        # search_xpath = "//a[contains(@class, 'filter__btn-applyFilters-js') and contains(text(), 'SEARCH')]"
        # search_btn = wait.until(EC.element_to_be_clickable((By.XPATH, search_xpath)))
        # driver.execute_script("arguments[0].click();", search_btn)
        
        # 5. Pagination and Download
        while True:
            time.sleep(3) # Wait for results to refresh
            download_pdfs(driver, ticker, base_dir)
            
            try:
                next_btn = driver.find_element(By.LINK_TEXT, "Next")
                if "disabled" in next_btn.get_attribute("class"):
                    break
                driver.execute_script("arguments[0].click();", next_btn)
                logger.info("Next page...")
            except:
                break

    except Exception as e:
        logger.error(f"Error for {ticker}: {e}")
        driver.save_screenshot(f"fail_{ticker}.png")
    finally:
        driver.quit()

if __name__ == "__main__":
    # Load your CSV here
    df = pd.read_csv("instruments.csv")
    tickers = df['instruments'].tolist()

    
    for t in tqdm(tickers):
        scrape_etp_prospectus(t, headless=True)