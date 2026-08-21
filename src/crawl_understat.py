"""
understat xG re-crawl v2 (2019/20 - 2025/26, top five leagues)
====================
- Endpoint: https://understat.com/getLeagueData/{league}/{year}
  (the API actually called by the page's league.min.js; requires the page session
  cookie first, XHR headers, and gzip response handling)
- Output: sci_redo/data/raw_understat/understat_{league}_{year}.csv
  (one row per team per match, same schema as the legacy understat_per_game.csv)
- Proxy: via the HTTP(S)_PROXY environment variable (run with a proxy enabled)
- Polite crawling: 1.5s between pages, up to 3 retries on failure
"""
import gzip
import json
import os
import time
import urllib.request
import http.cookiejar

import paths
OUT = paths.RAW_UNDERSTAT
os.makedirs(OUT, exist_ok=True)

LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
YEARS = list(range(2019, 2026))  # seasons 2019/20 ~ 2025/26

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Encoding": "gzip",
}


def make_opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj)), cj


def fetch(opener, url, headers, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=60) as resp:
                raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    retry {attempt+1}/{retries}: {e}")
            time.sleep(3 * (attempt + 1))
    return None


def rowify_teams(teams, league, year):
    """Expand teams[].history into one row per team per match (same schema as the legacy file)."""
    rows = []
    for tid, team in teams.items():
        name = team.get("title")
        for h in team.get("history", []):
            ppda = h.get("ppda") or {}
            oppda = h.get("ppda_allowed") or {}
            coef = (ppda["att"] / ppda["def"]) if ppda.get("def") else None
            ocoef = (oppda["att"] / oppda["def"]) if oppda.get("def") else None
            rows.append({
                "league": league, "year": year,
                "h_a": h.get("h_a"), "team": name,
                "xG": h.get("xG"), "xGA": h.get("xGA"),
                "npxG": h.get("npxG"), "npxGA": h.get("npxGA"),
                "deep": h.get("deep"), "deep_allowed": h.get("deep_allowed"),
                "scored": h.get("scored"), "missed": h.get("missed"),
                "xpts": h.get("xpts"), "result": h.get("result"),
                "date": h.get("date"),
                "wins": h.get("wins"), "draws": h.get("draws"),
                "loses": h.get("loses"), "pts": h.get("pts"),
                "npxGD": h.get("npxGD"),
                "ppda_coef": coef, "ppda_att": ppda.get("att"), "ppda_def": ppda.get("def"),
                "oppda_coef": ocoef, "oppda_att": oppda.get("att"), "oppda_def": oppda.get("def"),
            })
    return rows


def add_diff_cols(df):
    """Consistent with the legacy file: xG_diff = xG - team season mean; xpts_diff = xpts - actual points."""
    if df.empty:
        return df
    g = df.groupby(["league", "year", "team"])
    df["xG_diff"] = df["xG"] - g["xG"].transform("mean")
    df["xGA_diff"] = df["xGA"] - g["xGA"].transform("mean")
    pts_map = {"w": 3, "d": 1, "l": 0}
    df["xpts_diff"] = df["xpts"] - df["result"].map(pts_map)
    return df


def main():
    import pandas as pd
    opener, cj = make_opener()
    total = 0
    for league in LEAGUES:
        for year in YEARS:
            out_csv = os.path.join(OUT, f"understat_{league}_{year}.csv")
            if os.path.exists(out_csv):
                print(f"skip {league} {year} (already exists)")
                continue
            page_url = f"https://understat.com/league/{league}/{year}"
            data_url = f"https://understat.com/getLeagueData/{league}/{year}"
            headers = {**HEADERS, "Referer": page_url}
            print(f"fetch {league} {year} ...", flush=True)
            # Fetch the page first (to seed cookies)
            fetch(opener, page_url, headers)
            body = fetch(opener, data_url, headers)
            if not body:
                print(f"  FAILED {league} {year}")
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                print(f"  JSON FAILED {league} {year}: {e}")
                continue
            rows = rowify_teams(data.get("teams", {}), league, year)
            if not rows:
                print(f"  EMPTY {league} {year}")
                continue
            df = add_diff_cols(pd.DataFrame(rows))
            df.to_csv(out_csv, index=False)
            print(f"  {len(df)} rows -> {os.path.basename(out_csv)}")
            total += len(df)
            time.sleep(1.5)
    print(f"\ndone, {total} rows in total. Output dir: {OUT}")


if __name__ == "__main__":
    main()
