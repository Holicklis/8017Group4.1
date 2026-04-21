import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_full_etp_list(output_dir=None):
    # 1. Setup absolute path for the download directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "data", "ETF", "summary", "ETP_Data_Export.xlsx")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment once tested
    
    # Configure Chrome for automatic downloads
    prefs = {
        "download.default_directory": output_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    
    driver.maximize_window()

    wait = WebDriverWait(driver, 5)

    try:
        print("Navigating to HKEX ETP page...")
        driver.get("https://www.hkex.com.hk/Market-Data/Securities-Prices/Exchange-Traded-Products?sc_lang=en")

        # 2. Handle Cookie Banner
        try:
            cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_btn.click()
            print("Cookies accepted.")
        except:
            pass

        # 3. Click "Past 1 Day Data"
        print("Selecting 'Past 1 Day' period...")
        period_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.etps_period[data-period='d1']")))
        driver.execute_script("arguments[0].click();", period_btn)
        time.sleep(2) # Wait for table to refresh

        # 5. Handle the Modal Confirm (Using your specific HTML)
        print("Waiting for Date Range Modal...")
        # Targeting the div with data-ctl="confirm"
        # 5. Handle the Modal Confirm (Force Logic)
        print("Forcing Confirm in modal...")
        try:
            # We only wait for it to exist in the code (presence), not necessarily 'clickable'
            confirm_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.dateRangeCtl[data-ctl='confirm']")))
            
            # We use JavaScript to click it instantly. 
            # This works even if Selenium thinks the button is obscured or not ready.
            driver.execute_script("arguments[0].click();", confirm_btn)
            print("Confirm triggered via JS.")
        except Exception as e:
            print(f"Could not find confirm button: {e}")
            # Fallback: sometimes the button text is inside a span or capitalized differently
            driver.execute_script("$(\"div[data-ctl='confirm']\").click();")        
        print("Clicking 'Confirm' in modal...")
        driver.execute_script("arguments[0].click();", confirm_btn)
        
        print(f"Download triggered. Check directory: {output_dir}")
        
        # 6. Essential sleep to allow the download to finish before closing the browser
        # Generating XLSX files on the server can take a few seconds
        time.sleep(10) 

    except Exception as e:
        print(f"Error occurred: {e}")
        driver.save_screenshot("debug_master_list.png")
    finally:
        driver.quit()

if __name__ == "__main__":
    download_full_etp_list()