"""
xG feature augmentation (pipeline v3)
=====================================
Building on the leak-free v2 pipeline output, append understat xG time-series
features (likewise strictly available before kickoff):
- Team rolling means over the last 5 matches for xG/xGA/npxG/ppda/deep/xpts
  (shift excludes the current match)
- Home-away xG form difference (H_xg_roll5 - A_xg_roll5)

Team name mapping: football-data.co.uk abbreviations -> understat full names
(all teams in the top five leagues, 2019-2025).

Outputs (overwrite data/processed; v3 is the canonical version):
- all_matches_featurized.csv (with xG features; the v2 non-xG version is saved separately as _v2_noxg.csv)
- train/val/test_dataset.pkl + scaler.pkl + feature_cols.pkl
"""
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import paths
OUT = paths.PROCESSED
RAW_XG = paths.RAW_UNDERSTAT
ROLL_WIN = 5

# ---------------- Team name mapping: FD abbreviations -> understat full names ----------------
TEAM_MAP = {
    # EPL
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Newcastle": "Newcastle United", "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers", "West Brom": "West Bromwich Albion",
    # La Liga
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Betis": "Real Betis", "Celta": "Celta Vigo", "Espanol": "Espanyol",
    "Huesca": "SD Huesca", "Sociedad": "Real Sociedad",
    "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano",
    "Oviedo": "Real Oviedo",
    # Bundesliga
    "Dortmund": "Borussia Dortmund", "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne", "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Greuther Furth": "Greuther Fuerth", "Hamburg": "Hamburger SV",
    "Hertha": "Hertha Berlin", "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia M.Gladbach", "Mainz": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig", "St Pauli": "St. Pauli",
    "Stuttgart": "VfB Stuttgart", "Bielefeld": "Arminia Bielefeld",
    "Heidenheim": "FC Heidenheim",
    # Serie A
    "Milan": "AC Milan", "Parma": "Parma Calcio 1913", "Spal": "SPAL 2013",
    # Ligue 1
    "Paris SG": "Paris Saint Germain", "St Etienne": "Saint-Etienne",
    "Clermont": "Clermont Foot",
}
DIV_TO_XG_LEAGUE = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga",
                    "I1": "Serie_A", "F1": "Ligue_1"}


def load_understat():
    dfs = []
    for p in sorted(glob.glob(os.path.join(RAW_XG, "*.csv"))):
        d = pd.read_csv(p)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        dfs.append(d)
    u = pd.concat(dfs, ignore_index=True)
    u = u.dropna(subset=["date", "team"]).copy()
    # Normalize to date (FD dates are midnight 00:00; understat carries kickoff times)
    u["date"] = u["date"].dt.normalize()
    u = u.sort_values(["team", "date"]).reset_index(drop=True)
    return u


def team_rolling(u, col):
    """Sorted by (team, date); shift(1) excludes the current match, then rolling mean."""
    return u.groupby("team")[col].transform(
        lambda x: x.shift(1).rolling(ROLL_WIN, min_periods=1).mean())


def build_xg_features(u):
    cols = ["xG", "xGA", "npxG", "npxGA", "ppda_coef", "deep", "xpts"]
    out = {}
    for c in cols:
        out[c] = team_rolling(u, c)
    feat = pd.DataFrame(out)
    feat["team"] = u["team"].values
    feat["date"] = u["date"].values
    return feat


def main():
    v2_path = os.path.join(OUT, "all_matches_featurized_v2_noxg.csv")
    feat = pd.read_csv(v2_path, parse_dates=["Date"])
    # Drop any residual xG columns (regenerated if the v2 backup was contaminated)
    xg_cols = [c for c in feat.columns if c.startswith(("H_x", "A_x"))
               or c == "xg_diff_roll"]
    if xg_cols:
        feat = feat.drop(columns=xg_cols)
        print(f"cleaned residual xG columns: {xg_cols}")

    u = load_understat()
    xg = build_xg_features(u)
    xg["team_fd"] = xg["team"].map({v: k for k, v in TEAM_MAP.items()})
    xg["team_fd"] = xg["team_fd"].fillna(xg["team"])  # teams whose FD name matches directly
    print(f"understat rows: {len(u)}")
    print(f"unique team_fd values after mapping: {xg['team_fd'].nunique()}")

    # Merge: home team
    hm = xg[["team_fd", "date", "xG", "xGA", "npxG", "npxGA", "ppda_coef", "deep", "xpts"]].copy()
    hm = hm.rename(columns={"team_fd": "HomeTeam", "date": "Date"})
    hm.columns = ["HomeTeam", "Date"] + [f"H_{c}" for c in hm.columns[2:]]
    am = xg[["team_fd", "date", "xG", "xGA", "npxG", "npxGA", "ppda_coef", "deep", "xpts"]].copy()
    am = am.rename(columns={"team_fd": "AwayTeam", "date": "Date"})
    am.columns = ["AwayTeam", "Date"] + [f"A_{c}" for c in am.columns[2:]]

    n_before = len(feat)
    feat = feat.merge(hm, on=["HomeTeam", "Date"], how="left")
    feat = feat.merge(am, on=["AwayTeam", "Date"], how="left")
    print(f"rows before merge {n_before}, after merge {len(feat)} (should be the same)")

    matched_h = feat["H_xG"].notna().mean()
    matched_a = feat["A_xG"].notna().mean()
    print(f"home xG match rate: {matched_h:.3f}  away xG match rate: {matched_a:.3f}")

    # Home-away xG form difference
    feat["xg_diff_roll"] = feat["H_xG"] - feat["A_xG"]

    # Re-split + normalize (statistics fitted on train only)
    feature_cols = [c for c in feat.columns
                    if c not in ("Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y")
                    and feat[c].notna().sum() > 0]
    train = feat[feat["Date"] < "2024-08-01"]
    val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
    test = feat[feat["Date"] >= "2025-08-01"]

    medians = train[feature_cols].median()
    for part in (train, val, test):
        part.loc[:, feature_cols] = part[feature_cols].fillna(medians)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    def pack(X, part):
        return {"features": X, "target": part["y"].values.astype(int),
                "meta": part[["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR"]],
                "feature_cols": feature_cols}

    joblib.dump(pack(X_train, train), os.path.join(OUT, "train_dataset.pkl"))
    joblib.dump(pack(X_val, val), os.path.join(OUT, "val_dataset.pkl"))
    joblib.dump(pack(X_test, test), os.path.join(OUT, "test_dataset.pkl"))
    joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(OUT, "feature_cols.pkl"))
    feat.to_csv(os.path.join(OUT, "all_matches_featurized.csv"), index=False)

    print(f"v3 done: {len(feature_cols)} features (v2 had 45, {len(feature_cols) - 45} new xG features)")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
