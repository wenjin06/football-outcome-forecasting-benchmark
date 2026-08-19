"""
无泄漏数据管道 v2（SCI 重做版）
====================
原则：
1. 只用赛前可得信息构造特征（历史滚动状态、赛前积分排名、裁判历史、初盘/收盘赔率）
2. 严格时序划分：train < 2024-08-01 <= val < 2025-08-01 <= test
3. 任何统计量（scaler/填充）只 fit train
4. 明确排除：当前场统计、当前场结果、任何未来信息

v2 修复：球队特征基于"该队全部比赛"（主+客）统一构建，不再按主/客场分表。
输出：data/processed/{train,val,test}_dataset.pkl + all_matches_featurized.csv
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_FOLDER = r"E:\论文\structured_data"
OUT_FOLDER = r"E:\论文\sci_redo\data\processed"
ROLL_WIN = 5  # 滚动窗口场次

SEASON_START = {2019: "2019-08-01", 2020: "2020-08-01", 2021: "2021-08-01",
                2022: "2022-08-01", 2023: "2023-08-01", 2024: "2024-08-01",
                2025: "2025-08-01"}
VAL_START = "2024-08-01"
TEST_START = "2025-08-01"

# ============ 1. 读取全部 CSV ============
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

# ============ 2. 赛季标签 ============
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

# ============ 3. 球队状态特征（统一出场表：主+客全部比赛） ============
def team_features(df):
    """
    构建"队-出场"长表（每场比赛两条：主队一条、客队一条），按 (队, 时间) 排序后：
    - 近 ROLL_WIN 场进球/失球/积分/胜/射门/射正/角球/犯规均值（shift 排除当前场）
    - 赛季累计积分/净胜球/场次（赛前，按 队+赛季 分组）
    - 联赛排名（赛前，按积分+净胜球排序）
    返回 (home_feat, away_feat)，索引为 match_idx。
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

    # 赛季累计（赛前）：队+赛季 分组
    g2 = t.groupby(["team", "Season"])
    t["cum_pts"] = g2["Pts"].cumsum() - t["Pts"]
    t["cum_gf"] = g2["GF"].cumsum() - t["GF"]
    t["cum_ga"] = g2["GA"].cumsum() - t["GA"]
    t["cum_games"] = g2.cumcount()
    t["cum_gd"] = t["cum_gf"] - t["cum_ga"]
    t["ppg"] = (t["cum_pts"] / t["cum_games"].replace(0, np.nan)).values

    # 联赛排名（赛前）：同 (Div, Season) 内，该日期之前各队最后记录按积分+净胜球排序
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
            # 关键：Date 用目标比赛日 dt，而不是该队最后一场的日期
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

# ============ 4. 裁判历史特征 ============
def referee_features(df):
    """裁判历史：仅用该裁判过去执法场次。仅 E0(英超) 有 Referee 列，其余联赛为 NaN。"""
    if "Referee" not in df.columns or df["Referee"].isna().all():
        return pd.DataFrame(index=df.index)
    ref = df[["Date", "Referee", "HY", "AY", "HR", "AR", "HF", "AF"]].copy()
    # 保持 df 原始索引；mergesort 保证同 Date 行顺序稳定，索引不错位
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

# ============ 5. 赔率/市场特征 ============
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

# ============ 6. 主流程 ============
def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    print("[1/5] 读取全部 CSV...")
    df = load_all_csv(DATA_FOLDER)
    print(f"    合并后总场次: {len(df)}, 日期 {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    print("[2/5] 打赛季标签...")
    df = add_season(df)
    print(f"    赛季分布: {df['Season'].value_counts().sort_index().to_dict()}")

    print("[3/5] 构造特征（严格赛前）...")
    home_feat, away_feat = team_features(df)
    ref_feat = referee_features(df)
    odds_feat = odds_features(df)

    feats = df[["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    feats = feats.join(home_feat).join(away_feat).join(ref_feat).join(odds_feat)
    feats["y"] = feats["FTR"].map({"H": 0, "D": 1, "A": 2})

    drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
    feature_cols = [c for c in feats.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if feats[c].notna().sum() > 0]
    print(f"    特征数: {len(feature_cols)}")

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
    print(f"[4/5] 已保存至 {OUT_FOLDER}")
    print("[5/5] 完成。")

if __name__ == "__main__":
    main()
