"""Leak-free verification (manual recomputation against the raw data)"""
import pandas as pd, glob

raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(r"E:\论文\structured_data\*.csv"))], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"]).sort_values("Date").reset_index(drop=True)
feat = pd.read_csv(r"E:\论文\sci_redo\data\processed\all_matches_featurized.csv", parse_dates=["Date"])

def team_prior_matches(team, date, n=None):
    m = raw[(raw["Date"] < date) & ((raw["HomeTeam"] == team) | (raw["AwayTeam"] == team))]
    return m.tail(n) if n else m

print("=== check 1: H_roll_GF excludes the current match ===")
sample = raw[(raw["HomeTeam"] == "Liverpool") & (raw["Date"] >= "2025-08-01")].head(3)
for _, r in sample.iterrows():
    prior = team_prior_matches("Liverpool", r["Date"], 5)
    manual = prior.apply(lambda x: x["FTHG"] if x["HomeTeam"] == "Liverpool" else x["FTAG"], axis=1).mean()
    fr = feat[(feat["HomeTeam"] == "Liverpool") & (feat["Date"] == r["Date"])]
    if len(fr):
        fv = fr["H_roll_GF"].iloc[0]
        print(f"  {r['Date'].date()} vs {r['AwayTeam']}: feature={fv:.4f} manual={manual:.4f} {'OK' if abs(fv-manual)<1e-6 else 'LEAK!'}")

print("\n=== check 2: H_season_pts pre-match cumulative points (same season) ===")
r = raw[(raw["HomeTeam"] == "Arsenal") & (raw["Date"] >= "2025-08-01")].sort_values("Date").iloc[2]
prior = team_prior_matches("Arsenal", r["Date"])
prior = prior[prior["Date"] >= "2025-08-01"]
pts = 0
for _, x in prior.iterrows():
    ftr = x["FTR"]
    pts += 3 if (x["HomeTeam"] == "Arsenal" and ftr == "H") or (x["AwayTeam"] == "Arsenal" and ftr == "A") else (1 if ftr == "D" else 0)
fr = feat[(feat["HomeTeam"] == "Arsenal") & (feat["Date"] == r["Date"])]
if len(fr):
    fv = fr["H_season_pts"].iloc[0]
    print(f"  {r['Date'].date()} vs {r['AwayTeam']}: feature={fv} manual={pts} {'OK' if fv==pts else 'LEAK!'}")

print("\n=== check 3: rank = points-based rank before the date (within league) ===")
r = raw[(raw["HomeTeam"] == "Arsenal") & (raw["Date"] >= "2025-08-01")].sort_values("Date").iloc[4]
before = raw[(raw["Date"] < r["Date"]) & (raw["Date"] >= "2025-08-01") & (raw["Div"] == r["Div"])]
rows = []
for team in pd.unique(before[["HomeTeam", "AwayTeam"]].values.ravel()):
    sub = before[(before["HomeTeam"] == team) | (before["AwayTeam"] == team)]
    pts = 0; gf = 0; ga = 0
    for _, x in sub.iterrows():
        ftr = x["FTR"]
        if x["HomeTeam"] == team:
            pts += 3 if ftr == "H" else (1 if ftr == "D" else 0); gf += x["FTHG"]; ga += x["FTAG"]
        else:
            pts += 3 if ftr == "A" else (1 if ftr == "D" else 0); gf += x["FTAG"]; ga += x["FTHG"]
    rows.append((team, pts, gf - ga))
rows.sort(key=lambda t: (-t[1], -t[2]))
rank = [t[0] for t in rows].index("Arsenal") + 1
fr = feat[(feat["HomeTeam"] == "Arsenal") & (feat["Date"] == r["Date"])]
if len(fr):
    fv = fr["H_rank"].iloc[0]
    print(f"  {r['Date'].date()} vs {r['AwayTeam']}: feature={fv} manual={rank} {'OK' if fv==rank else 'LEAK!'}")

print("\n=== check 4: referee features (only EPL has Referee) ===")
feat2 = feat.merge(raw[["Date", "HomeTeam", "AwayTeam", "Referee"]], on=["Date", "HomeTeam", "AwayTeam"], how="left")
raw_e0 = raw[raw["Div"] == "E0"]
r = raw_e0[raw_e0["Referee"].notna()].iloc[2000]
ref_prior = raw_e0[(raw_e0["Referee"] == r["Referee"]) & (raw_e0["Date"] < r["Date"])]
manual = (ref_prior["HY"] + ref_prior["AY"] + ref_prior["HR"] + ref_prior["AR"]).mean()
fr = feat2[(feat2["Referee"] == r["Referee"]) & (feat2["Date"] == r["Date"])]
if len(fr):
    fv = fr["ref_cards_avg"].iloc[0]
    print(f"  referee {r['Referee']} {r['Date'].date()}: feature={fv:.4f} manual={manual:.4f} {'OK' if abs(fv-manual)<1e-6 else 'LEAK!'}")
else:
    print("  no matching feature row found")
print("  E0 matches:", len(raw_e0), " E0 Referee non-null:", raw_e0["Referee"].notna().sum())

print("\n=== check 5: market implied probabilities sum ===")
s = feat[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].sum(axis=1)
print(f"  min={s.min():.6f} max={s.max():.6f} median={s.median():.6f} (should be ~1)")
print(f"  missing rows: {feat['mkt_prob_H'].isna().sum()} / {len(feat)}")

print("\n=== check 6: feature NaN ratios Top5 ===")
cols = [c for c in feat.columns if c not in ("Div","Date","Season","HomeTeam","AwayTeam","FTR","y")]
print(feat[cols].isna().mean().sort_values(ascending=False).head(5).to_string())
