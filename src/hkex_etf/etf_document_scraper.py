import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return project root based on this file location."""
    return Path(__file__).resolve().parents[2]


def _default_log_file() -> Path:
    """Return default log file location for document scraping."""
    return _project_root() / "logs" / "etp_prospectus.log"


def configure_logging(log_file: Optional[Union[str, Path]] = None) -> None:
    """Configure stream + file logging for this module."""
    target_log = Path(log_file).expanduser().resolve() if log_file else _default_log_file()
    target_log.parent.mkdir(parents=True, exist_ok=True)

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(target_log), logging.StreamHandler()],
    )


def setup_driver(headless: bool = False) -> webdriver.Chrome:
    """Create Selenium Chrome driver for HKEX document scraping."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=chrome_options)


def _safe_pdf_title(raw_title: str, max_length: int = 60) -> str:
    """Return file-system-safe PDF title segment joined by underscores."""
    clean_title = "".join(ch for ch in raw_title if ch.isalnum() or ch in " -_").strip()
    if not clean_title:
        return "document"
    normalized = "_".join(clean_title.replace("-", " ").split())
    return normalized[:max_length]


def _ticker_output_dir(ticker: Union[str, int]) -> Path:
    """Return standard output directory for one ticker's PDFs."""
    data_dir = _project_root() / "data"
    lower_root = data_dir / "etf"
    legacy_roots = [data_dir / "ETF"]
    etf_root = lower_root
    if not etf_root.exists():
        for path in legacy_roots:
            if path.exists():
                etf_root = path
                break
    return etf_root / "documentation" / str(ticker).zfill(5) / "pdf"


def download_pdfs(driver: webdriver.Chrome, ticker: Union[str, int], folder: Path) -> int:
    """Extract and save PDF links from current HKEX News result page."""
    download_count = 0
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "table")))
        pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
        logger.info("[%s] Found %s document(s) on this page", ticker, len(pdf_links))

        for link in pdf_links:
            url = link.get_attribute("href")
            raw_text = link.text.split("(")[0].strip()
            title = _safe_pdf_title(raw_text)
            filename = f"{str(ticker).zfill(5)}_{title}.pdf"
            filepath = folder / filename
            if filepath.exists():
                logger.info("[%s] Skipped existing file: %s", ticker, filename)
                continue
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                if response.status_code == 200:
                    filepath.write_bytes(response.content)
                    logger.info("[%s] Saved: %s", ticker, filename)
                    download_count += 1
                else:
                    logger.warning("[%s] Download returned %s: %s", ticker, response.status_code, url)
            except Exception as exc:
                logger.error("[%s] Failed %s: %s", ticker, url, exc)
    except Exception:
        logger.info("[%s] No document table found on current page", ticker)
    return download_count


def scrape_etp_prospectus(ticker: Union[str, int], headless: bool = False) -> int:
    """
    Scrape and download all available HKEX prospectus documents for one ticker.

    Returns:
        Number of PDFs downloaded.
    """
    formatted_ticker = str(ticker).lstrip("0")
    output_dir = _ticker_output_dir(ticker)
    output_dir.mkdir(parents=True, exist_ok=True)

    driver = setup_driver(headless=headless)
    wait = WebDriverWait(driver, 15)
    total_downloads = 0

    try:
        url = (
            "https://www.hkex.com.hk/Market-Data/Securities-Prices/Exchange-Traded-Products/"
            f"Exchange-Traded-Products-Quote?sym={formatted_ticker}&sc_lang=en"
        )
        logger.info("[%s] Navigating to quote page: %s", ticker, url)
        driver.get(url)

        logger.info("[%s] Triggering prospectus redirect", ticker)
        prospectus_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.sublink.prospectus h2 a")))
        driver.execute_script("arguments[0].click();", prospectus_link)

        time.sleep(3)
        if len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])

        while True:
            time.sleep(3)
            total_downloads += download_pdfs(driver, ticker, output_dir)
            try:
                next_btn = driver.find_element(By.LINK_TEXT, "Next")
                if "disabled" in next_btn.get_attribute("class"):
                    break
                driver.execute_script("arguments[0].click();", next_btn)
                logger.info("[%s] Next page", ticker)
            except Exception:
                break
    except Exception as exc:
        logger.error("[%s] Error: %s", ticker, exc)
        driver.save_screenshot(f"fail_{ticker}.png")
    finally:
        driver.quit()
    return total_downloads


def load_tickers_from_csv(csv_path: Union[str, Path], column: str = "instruments") -> List[str]:
    """Load ticker values from a CSV file."""
    df_tickers = pd.read_csv(csv_path)
    if column not in df_tickers.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")
    return [str(value) for value in df_tickers[column].dropna().tolist()]


def scrape_many_tickers(tickers: Sequence[Union[str, int]], headless: bool = True) -> None:
    """Run document scraping for multiple tickers."""
    for ticker in tqdm(tickers, desc="Scraping ETF documents"):
        scrape_etp_prospectus(ticker=ticker, headless=headless)


if __name__ == "__main__":
    configure_logging()
    default_csv = _project_root() / "data" / "etf" / "instruments" / "all_hkd_etf.csv"
    if default_csv.exists():
        ticker_values = load_tickers_from_csv(default_csv, column="instruments")
        scrape_many_tickers(ticker_values, headless=True)
    else:
        logger.error("Instrument file not found: %s", default_csv)