import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

HKEX_ETP_URL = "https://www.hkex.com.hk/Market-Data/Securities-Prices/Exchange-Traded-Products?sc_lang=en"
DEFAULT_FILENAME = "ETP_Data_Export.xlsx"


def _project_root() -> Path:
    """Return project root based on this file location."""
    return Path(__file__).resolve().parents[2]


def _default_summary_dir() -> Path:
    """Return default directory where HKEX summary export is stored."""
    data_dir = _project_root() / "data"
    lower_path = data_dir / "etf" / "summary"
    legacy_paths = [data_dir / "ETF" / "Summary", data_dir / "ETF" / "summary"]
    if lower_path.exists():
        return lower_path
    for path in legacy_paths:
        if path.exists():
            return path
    return lower_path


def _default_instruments_csv_path() -> Path:
    """Return default output CSV path for ETF instruments list."""
    return _project_root() / "data" / "etf" / "instruments" / "etf.csv"


def _resolve_download_dir(output_dir: Optional[Union[str, Path]]) -> Path:
    """
    Resolve target download directory.

    Backward compatible behavior:
    - If `output_dir` points to a file path, use its parent directory.
    - If omitted, use `data/etf/summary`.
    """
    if output_dir is None:
        return _default_summary_dir()

    path_output = Path(output_dir).expanduser().resolve()
    if path_output.suffix:
        return path_output.parent
    return path_output


def export_hkd_etf_instruments(
    summary_xlsx: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export unique, sorted HKD ETF stock codes to CSV.

    Logic follows notebook filter:
    - Stock code* < 8000
    - Base currency* == "HKD"
    """
    xlsx_path = Path(summary_xlsx).expanduser().resolve()
    target_csv = (
        Path(output_csv).expanduser().resolve() if output_csv else _default_instruments_csv_path()
    )
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path, skipfooter=2)
    filtered = df.query("`Stock code*` < 8000 and `Base currency*` == 'HKD'")
    instruments = (
        filtered["Stock code*"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .rename("instruments")
        .to_frame()
    )
    instruments.to_csv(target_csv, index=False)
    logger.info("Exported %s instruments to %s", len(instruments), target_csv)
    return target_csv


def _build_driver(download_dir: Path, headless: bool = False) -> Any:
    """Create a Selenium Chrome driver configured for auto download."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=chrome_options)


def download_full_etp_list(
    output_dir: Optional[Union[str, Path]] = None,
    timeout_seconds: int = 20,
    headless: bool = False,
) -> Optional[Path]:
    """
    Download the latest HKEX ETP data export file.

    Args:
        output_dir: Download directory (or file path; parent will be used).
        timeout_seconds: UI wait timeout for Selenium interactions.
        headless: Whether to run Chrome in headless mode.

    Returns:
        Path of downloaded XLSX file if discovered, otherwise None.
    """
    target_dir = _resolve_download_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_xlsx = {p.name for p in target_dir.glob("*.xlsx")}

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    driver = _build_driver(target_dir, headless=headless)
    wait = WebDriverWait(driver, timeout_seconds)

    try:
        logger.info("Opening HKEX ETP page")
        driver.get(HKEX_ETP_URL)

        try:
            cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_btn.click()
            logger.info("Cookie banner accepted")
        except Exception:
            logger.debug("Cookie banner not present or not clickable")

        logger.info("Selecting 'Past 1 Day'")
        period_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.etps_period[data-period='d1']"))
        )
        driver.execute_script("arguments[0].click();", period_btn)
        time.sleep(2)

        logger.info("Confirming modal to trigger export")
        confirm_btn = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.dateRangeCtl[data-ctl='confirm']"))
        )
        driver.execute_script("arguments[0].click();", confirm_btn)
        time.sleep(8)

        new_xlsx = sorted(
            [p for p in target_dir.glob("*.xlsx") if p.name not in existing_xlsx],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if new_xlsx:
            logger.info("Download completed: %s", new_xlsx[0])
            return new_xlsx[0]

        fallback = sorted(target_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if fallback:
            logger.warning("No newly created xlsx detected; latest file is %s", fallback[0])
            return fallback[0]

        logger.warning("No xlsx file found in download directory: %s", target_dir)
        return None
    except Exception as exc:
        logger.exception("Failed to download HKEX ETP export: %s", exc)
        screenshot_path = _project_root() / "debug_master_list.png"
        driver.save_screenshot(str(screenshot_path))
        logger.info("Saved debug screenshot to %s", screenshot_path)
        return None
    finally:
        driver.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    downloaded_file = download_full_etp_list()
    if downloaded_file is not None:
        export_hkd_etf_instruments(downloaded_file)