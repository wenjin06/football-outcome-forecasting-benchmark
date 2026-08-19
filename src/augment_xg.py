"""
xG 特征增强（管道 v3）
====================
在 v2 无泄漏管道产物基础上，追加 understat xG 时序特征（同样严格赛前可得）：
- 球队近 5 场 xG/xGA/npxG/ppda压迫强度/deep推进/xpts 均值（shift 排除当前场）
- 主客 xG 状态差（H_xg_roll5 - A_xg_roll5）

队名映射：football-data.co.uk 简称 -> understat 全称（五大联赛，2019-2025 全部球队）。

输出（覆盖 data/processed，v3 为规范版）：
- all_matches_featurized.csv（含 xG 特征；v2 无 xG 版另存 _v2_noxg.csv）
- train/val/test_dataset.pkl + scaler.pkl + feature_cols.pkl
"""
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OUT = r"E:\论文\sci_redo\data\processed"
RAW_XG = r"E:\论文\sci_redo\data\raw_understat"
ROLL_WIN = 5

# ---------------- 队名映射：FD 简称 -> understat 全称 ----------------
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
    # 归一化到日期（FD 日期为午夜 00:00，understat 带开球时间）
    u["date"] = u["date"].dt.normalize()
    u = u.sort_values(["team", "date"]).reset_index(drop=True)
    return u


def team_rolling(u, col):
    """按 (team, date) 排序后，shift(1) 排除当前场，rolling 均值。"""
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
    # 清理可能残留的 xG 列（若 v2 备份曾被污染则重新生成）
    xg_cols = [c for c in feat.columns if c.startswith(("H_x", "A_x"))
               or c == "xg_diff_roll"]
    if xg_cols:
        feat = feat.drop(columns=xg_cols)
        print(f"已清理残留 xG 列: {xg_cols}")

    u = load_understat()
    xg = build_xg_features(u)
    xg["team_fd"] = xg["team"].map({v: k for k, v in TEAM_MAP.items()})
    xg["team_fd"] = xg["team_fd"].fillna(xg["team"])  # 未映射但同名的情况
    print(f"understat rows: {len(u)}")
    print(f"映射后 team_fd 唯一值: {xg['team_fd'].nunique()}")

    # 联表：主队
    hm = xg[["team_fd", "date", "xG", "xGA", "npxG", "npxGA", "ppda_coef", "deep", "xpts"]].copy()
    hm = hm.rename(columns={"team_fd": "HomeTeam", "date": "Date"})
    hm.columns = ["HomeTeam", "Date"] + [f"H_{c}" for c in hm.columns[2:]]
    am = xg[["team_fd", "date", "xG", "xGA", "npxG", "npxGA", "ppda_coef", "deep", "xpts"]].copy()
    am = am.rename(columns={"team_fd": "AwayTeam", "date": "Date"})
    am.columns = ["AwayTeam", "Date"] + [f"A_{c}" for c in am.columns[2:]]

    n_before = len(feat)
    feat = feat.merge(hm, on=["HomeTeam", "Date"], how="left")
    feat = feat.merge(am, on=["AwayTeam", "Date"], how="left")
    print(f"联表前 {n_before} 行，联表后 {len(feat)} 行（应为同一行数）")

    matched_h = feat["H_xG"].notna().mean()
    matched_a = feat["A_xG"].notna().mean()
    print(f"主队 xG 匹配率: {matched_h:.3f}  客队 xG 匹配率: {matched_a:.3f}")

    # 主客 xG 状态差
    feat["xg_diff_roll"] = feat["H_xG"] - feat["A_xG"]

    # 重新划分 + 归一化（统计量只 fit train）
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

    print(f"v3 完成：特征数 {len(feature_cols)}（v2 为 45，新增 xG 特征 {len(feature_cols) - 45} 个）")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
