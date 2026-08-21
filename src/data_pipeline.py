"""
Leak-free data pipeline v2 (SCI redo)
=====================================
Principles:
1. Features are built only from pre-match information (historical rolling form,
   pre-match points and standings, referee history, opening/closing odds)
2. Strict temporal split: train < 2024-08-01 <= val < 2025-08-01 <= test
3. Any statistics (scaler/imputation) are fitted on train only
4. Explicitly excluded: current-match statistics, current-match results, and any
   future information

v2 fix: team features are built from all matches of a team (home + away) in one
unified table, no longer split into separate home/away tables.
Output: data/processed/{train,val,test}_dataset.pkl + all_matches_featurized.csv
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import paths
DATA_FOLDER = paths.raw_data_dir()
OUT_FOLDER = paths.PROCESSED
ROLL_WIN = 5  # rolling window size in matches

SEASON_START = {2019: "2019-08-01", 2020: "2020-08-01", 2021: "2021-08-01",
                2022: "2022-08-01", 2023: "2023-08-01", 2024: "2024-08-01",
                2025: "2025-08-01"}
VAL_START = "2024-08-01"
TEST_START = "2025-08-01"

# ============ 1. Load all CSVs ============
def load_all_csv(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["_src"] = os.path.basename(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ============ 2. Season labels ============
def add_season(df):
    def season_of(d):
        for y in sorted(SEASON_START, reverse=True):
            if d >= pd.Timestamp(SEASON_START[y]):
                return y
        return None
    df["Season"] = df["Date"].apply(season_of)
    df = df.dropna(subset=["Season"]).copy()
    df["Season"] = df["Season"].astype(int)
    return df

# ============ 3. Team form features (unified appearance table: all home+away matches) ============
def team_features(df):
    """
    Build a team-appearance long table (two rows per match: one home, one away),
    sorted by (team, time):
    - Rolling means over the last ROLL_WIN matches of goals for/against, points,
      wins, shots, shots on target, corners, fouls (shift excludes the current match)
    - Season cumulative points/goal difference/games (pre-match, grouped by team + season)
    - League rank (pre-match, ordered by points + goal difference)
    Returns (home_feat, away_feat) indexed by match_idx.
    """
    rows = []
    for _, r in df.iterrows():
        for side, team in [("home", r["HomeTeam"]), ("away", r["AwayTeam"])]:
            gf = r["FTHG"] if side == "home" else r["FTAG"]
            ga = r["FTAG"] if side == "home" else r["FTHG"]
            ftr_win = (r["FTR"] == "H" and side == "home") or (r["FTR"] == "A" and side == "away")
            pts = 3 if ftr_win else (1 if r["FTR"] == "D" else 0)
            rows.append({
                "match_idx": r.name, "team": team, "side": side,
                "Div": r["Div"], "Season": r["Season"], "Date": r["Date"],
                "GF": gf, "GA": ga, "Pts": pts, "Win": int(ftr_win),
                "Shots": r["HS"] if side == "home" else r["AS"],
                "ShotsT": r["HST"] if side == "home" else r["AST"],
                "Corners": r["HC"] if side == "home" else r["AC"],
                "Fouls": r["HF"] if side == "home" else r["AF"],
            })
    t = pd.DataFrame(rows)
    t = t.sort_values(["team", "Date", "match_idx"]).reset_index(drop=True)

    g = t.groupby("team")
    for col in ["GF", "GA", "Pts", "Win", "Shots", "ShotsT", "Corners", "Fouls"]:
        t[f"{col}_prev"] = g[col].shift(1)
    roll_cols = ["GF", "GA", "Pts", "Win", "Shots", "ShotsT", "Corners", "Fouls"]
    roll = t.groupby("team")[[f"{c}_prev" for c in roll_cols]].transform(
        lambda x: x.rolling(ROLL_WIN, min_periods=1).mean())

    # Season cumulative (pre-match): grouped by team + season
    g2 = t.groupby(["team", "Season"])
    t["cum_pts"] = g2["Pts"].cumsum() - t["Pts"]
    t["cum_gf"] = g2["GF"].cumsum() - t["GF"]
    t["cum_ga"] = g2["GA"].cumsum() - t["GA"]
    t["cum_games"] = g2.cumcount()
    t["cum_gd"] = t["cum_gf"] - t["cum_ga"]
    t["ppg"] = (t["cum_pts"] / t["cum_games"].replace(0, np.nan)).values

    # League rank (pre-match): within each (Div, Season), rank each team by its last
    # record before the match date, ordered by points + goal difference
    rank_map = []
    for (div, season), sub in t.groupby(["Div", "Season"]):
        dates = np.sort(sub["Date"].unique())
        sub_sorted = sub.sort_values("Date")
        for dt in dates:
            before = sub_sorted[sub_sorted["Date"] < dt]
            if before.empty:
                continue
            last = before.groupby("team").tail(1).sort_values(["cum_pts", "cum_gd"], ascending=False)
            last = last.reset_index(drop=True)
            last["rank"] = np.arange(1, len(last) + 1)
            # Key: use the target match date dt, not the date of the team's last match
            rank_map.append(pd.DataFrame({
                "Div": div, "Season": season, "Date": dt,
                "team": last["team"].values, "rank": last["rank"].values,
            }))
    if rank_map:
        rm = pd.concat(rank_map, ignore_index=True)
        t = t.merge(rm, on=["Div", "Season", "Date", "team"], how="left")

    feat_cols = {
        "roll_GF": "GF", "roll_GA": "GA", "roll_Pts": "Pts", "roll_Win": "Win",
        "roll_Shots": "Shots", "roll_ShotsT": "ShotsT", "roll_Corners": "Corners", "roll_Fouls": "Fouls",
    }
    out = {}
    for name, c in feat_cols.items():
        out[name] = roll[f"{c}_prev"].values
    out["season_pts"] = t["cum_pts"].values
    out["season_gf"] = t["cum_gf"].values
    out["season_ga"] = t["cum_ga"].values
    out["season_games"] = t["cum_games"].values
    out["season_gd"] = t["cum_gd"].values
    out["season_ppg"] = t["ppg"].values
    if "rank" in t.columns:
        out["rank"] = t["rank"].values
    else:
        out["rank"] = np.full(len(t), np.nan)

    feat = pd.DataFrame(out)
    feat["match_idx"] = t["match_idx"].values
    feat["side"] = t["side"].values
    feat = feat.drop_duplicates(["match_idx", "side"]).set_index("match_idx")
    home = feat[feat["side"] == "home"].drop(columns=["side"])
    away = feat[feat["side"] == "away"].drop(columns=["side"])
    home.columns = ["H_" + c for c in home.columns]
    away.columns = ["A_" + c for c in away.columns]
    return home, away

# ============ 4. Referee history features ============
def referee_features(df):
    """Referee history: only matches previously officiated by the same referee are used. Only E0 (Premier League) has a Referee column; other leagues are NaN."""
    if "Referee" not in df.columns or df["Referee"].isna().all():
        return pd.DataFrame(index=df.index)
    ref = df[["Date", "Referee", "HY", "AY", "HR", "AR", "HF", "AF"]].copy()
    # Preserve the original df index; mergesort keeps row order stable within the same Date so indices do not shift
    ref = ref.sort_values("Date", kind="mergesort")
    ref["cards"] = ref["HY"] + ref["AY"] + ref["HR"] + ref["AR"]
    ref["reds"] = ref["HR"] + ref["AR"]
    ref["fouls"] = ref["HF"] + ref["AF"]
    g = ref.groupby("Referee", dropna=True)
    out = pd.DataFrame({
        "ref_cards_avg": g["cards"].transform(lambda x: x.shift(1).expanding().mean()),
        "ref_reds_avg": g["reds"].transform(lambda x: x.shift(1).expanding().mean()),
        "ref_fouls_avg": g["fouls"].transform(lambda x: x.shift(1).expanding().mean()),
        "ref_games": g["cards"].transform(lambda x: x.shift(1).expanding().count()),
    }, index=ref.index)
    return out

# ============ 5. Odds/market features ============
def odds_features(df):
    close_h = [c for c in df.columns if c.endswith("CH") and c not in ("B365CH",)
               and df[c].notna().sum() > 1000]
    out = pd.DataFrame(index=df.index)
    oc = df[["B365CH", "B365CD", "B365CA"]].copy()
    if "AvgCH" in df.columns:
        oc = oc.fillna(df[["AvgCH", "AvgCD", "AvgCA"]])
    inv = 1.0 / oc.replace(0, np.nan)
    s = inv.sum(axis=1)
    for i, k in enumerate(["H", "D", "A"]):
        out[f"mkt_prob_{k}"] = (inv.iloc[:, i] / s).values
    out["mkt_prob_max"] = out[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].max(axis=1)

    out["odds_move_H"] = (df["B365CH"] - df["B365H"]).values
    out["odds_move_D"] = (df["B365CD"] - df["B365D"]).values
    out["odds_move_A"] = (df["B365CA"] - df["B365A"]).values

    out["close_vol"] = df[close_h].std(axis=1).values if close_h else np.nan
    out["close_avg_h"] = df[close_h].mean(axis=1).values if close_h else np.nan
    out["open_avg_h"] = df["AvgH"].values if "AvgH" in df.columns else np.nan
    out["close_max_h"] = df[close_h].max(axis=1).values if close_h else np.nan
    return out

# ============ 6. Main pipeline ============
def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    print("[1/5] loading all CSVs...")
    df = load_all_csv(DATA_FOLDER)
    print(f"    total matches after merge: {len(df)}, dates {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    print("[2/5] adding season labels...")
    df = add_season(df)
    print(f"    season distribution: {df['Season'].value_counts().sort_index().to_dict()}")

    print("[3/5] building features (strictly pre-match)...")
    home_feat, away_feat = team_features(df)
    ref_feat = referee_features(df)
    odds_feat = odds_features(df)

    feats = df[["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    feats = feats.join(home_feat).join(away_feat).join(ref_feat).join(odds_feat)
    feats["y"] = feats["FTR"].map({"H": 0, "D": 1, "A": 2})

    drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
    feature_cols = [c for c in feats.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if feats[c].notna().sum() > 0]
    print(f"    feature count: {len(feature_cols)}")

    train = feats[feats["Date"] < VAL_START]
    val = feats[(feats["Date"] >= VAL_START) & (feats["Date"] < TEST_START)]
    test = feats[feats["Date"] >= TEST_START]
    print(f"    train={len(train)} val={len(val)} test={len(test)}")

    medians = train[feature_cols].median()
    for part in (train, val, test):
        part.loc[:, feature_cols] = part[feature_cols].fillna(medians)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    def pack(X, part):
        return {
            "features": X, "target": part["y"].values.astype(int),
            "meta": part[["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR"]],
            "feature_cols": feature_cols,
        }

    joblib.dump(pack(X_train, train), os.path.join(OUT_FOLDER, "train_dataset.pkl"))
    joblib.dump(pack(X_val, val), os.path.join(OUT_FOLDER, "val_dataset.pkl"))
    joblib.dump(pack(X_test, test), os.path.join(OUT_FOLDER, "test_dataset.pkl"))
    joblib.dump(scaler, os.path.join(OUT_FOLDER, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(OUT_FOLDER, "feature_cols.pkl"))
    feats.to_csv(os.path.join(OUT_FOLDER, "all_matches_featurized.csv"), index=False)
    print(f"[4/5] saved to {OUT_FOLDER}")
    print("[5/5] done.")

if __name__ == "__main__":
    main()
