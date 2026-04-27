from __future__ import annotations

import argparse
import calendar
import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

YEARS_ARCHIVE_URL = "https://www.wsj.com/news/archive/years"
MONTH_LINK_RE = re.compile(r"/news/archive/(\d{4})/([a-z]+)")
HEADLINE_PAIR_RE = re.compile(
    r'"articleUrl":"(https://www\.wsj\.com/[^"]+)","bylineData":\[[^\]]*?\],"headline":"([^"]+)"'
)

MONTH_NAME_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class ArchiveMonth:
    year: int
    month: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_logging(level: int = logging.INFO) -> Path:
    log_dir = _project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scrape_wsj_archive_titles.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    return log_file


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


def _discover_months(session: requests.Session) -> list[ArchiveMonth]:
    resp = session.get(YEARS_ARCHIVE_URL, headers=_headers(), timeout=30)
    resp.raise_for_status()
    month_pairs = {
        (int(year), MONTH_NAME_TO_NUM[name.lower()])
        for year, name in MONTH_LINK_RE.findall(resp.text)
        if name.lower() in MONTH_NAME_TO_NUM
    }
    out = [ArchiveMonth(year=y, month=m) for y, m in sorted(month_pairs)]
    if not out:
        raise RuntimeError("No archive month links found on years page.")
    return out


def _iter_dates(months: Iterable[ArchiveMonth]) -> list[date]:
    dates: list[date] = []
    for item in months:
        max_day = calendar.monthrange(item.year, item.month)[1]
        for day in range(1, max_day + 1):
            dates.append(date(item.year, item.month, day))
    return dates


def _day_url(d: date) -> str:
    return f"https://www.wsj.com/news/archive/{d.year:04d}/{d.month:02d}/{d.day:02d}"


def _fetch_day_titles(session: requests.Session, d: date, retries: int = 2) -> tuple[date, list[dict[str, str]], int]:
    url = _day_url(d)
    last_status = 0
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, headers=_headers(), timeout=30)
            last_status = resp.status_code
            if resp.status_code != 200:
                if attempt < retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                return d, [], last_status

            rows = []
            for article_url, headline in HEADLINE_PAIR_RE.findall(resp.text):
                clean_title = html.unescape(headline).strip()
                if not clean_title:
                    continue
                rows.append(
                    {
                        "date": d.isoformat(),
                        "title": clean_title,
                        "article_url": article_url,
                        "archive_day_url": url,
                    }
                )
            return d, rows, last_status
        except requests.RequestException:
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
            else:
                return d, [], last_status
    return d, [], last_status


def scrape_wsj_archive_titles(
    output_csv: Path,
    max_workers: int = 8,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, object]:
    logger.info("Starting WSJ archive title scrape")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        months = _discover_months(session)
        if start_year is not None:
            months = [m for m in months if m.year >= start_year]
        if end_year is not None:
            months = [m for m in months if m.year <= end_year]
        dates = _iter_dates(months)

        all_rows: list[dict[str, str]] = []
        statuses: dict[int, int] = {}
        processed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_day_titles, session, d) for d in dates]
            for fut in as_completed(futures):
                d, rows, status = fut.result()
                processed += 1
                statuses[status] = statuses.get(status, 0) + 1
                all_rows.extend(rows)
                if processed % 200 == 0:
                    logger.info("Processed %s/%s days | titles collected=%s", processed, len(dates), len(all_rows))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "title", "article_url"]).sort_values(["date", "title"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output_csv),
        "rows": int(len(df)),
        "months_scanned": int(len(months)),
        "days_scanned": int(len(dates)),
        "status_counts": statuses,
    }
    logger.info("Scrape completed. Output rows=%s saved to %s", summary["rows"], output_csv)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape WSJ archive headlines from day archive pages.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "news" / "wsj_archive_titles.csv",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    log_file = configure_logging()
    args = _parse_args()
    result = scrape_wsj_archive_titles(
        output_csv=args.output_csv,
        max_workers=max(args.max_workers, 1),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    logger.info("Run result: %s", result)
    logger.info("Logs saved to %s", log_file)


if __name__ == "__main__":
    main()
