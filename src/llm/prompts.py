"""
LLM 提示词模板（诚实版）
====================
设计原则：
1. 输入只包含赛前可得特征（与数据管道一致），显式声明信息截止时间，禁止事后信息
2. 系统提示注入真实领域知识：收盘赔率 de-vig 概率是极强的基线（市场有效性），
   模型必须给出超越市场的可论证理由，而不是默认自己比市场聪明
3. 输出固定 JSON：{"probs": [pH, pD, pA], "reasoning": "..."}
   推理必须可验证（引用给定特征数值），禁止编造新闻/伤病/阵容信息
4. 不联网、无检索：这就是"领域增强 + CoT"的真实边界，论文中如实描述
"""
import json

SYSTEM_PROMPT = """You are a football match outcome forecasting model used in an academic study.

DOMAIN KNOWLEDGE (verified facts, use them as priors):
- You receive ONLY pre-match information. The match result is UNKNOWN to you.
- Closing odds from major bookmakers, after removing the bookmaker margin (de-vig),
  are a very strong probability baseline. Bookmakers are highly efficient; beating
  them requires specific, defensible reasoning. Do not assume you are smarter than
  the market without concrete justification.
- In the top-5 European leagues, empirical outcome frequencies are approximately:
  home win 43%, away win 32%, draw 25%. Draw is the least frequent class.
- A larger ranking/strength gap between teams makes the match MORE predictable
  (favorite wins more often), not less.
- Historical referee tendencies (cards/fouls) have weak but measurable association
  with match dynamics; do not over-weight them.

TASK:
Given the pre-match feature card for one match, produce calibrated probabilities
for the three outcomes: HOME WIN, DRAW, AWAY WIN.

RULES:
- Reason step by step in the "reasoning" field: (1) compare team form and rank,
  (2) assess market signal (de-vig implied probabilities and odds movement),
  (3) consider referee context, (4) state your final adjustment relative to the
  market baseline and justify it with the given numbers.
- If you have no strong reason to deviate from the de-vig market probabilities,
  stay close to them. This is expected and correct behavior.
- NEVER invent news, injuries, lineups, or any information not present in the card.
- Output ONLY valid JSON: {"probs": [home_prob, draw_prob, away_prob], "reasoning": "..."}
  probs must be non-negative and sum to 1.
"""


def build_feature_card(feat_row, mode="full"):
    """把一行赛前特征整理成文本卡片（只含赛前可得项）。
    mode: full | market | stats | market_stats
      - full: HOME/AWAY + MARKET + REFEREE（默认，论文主实验）
      - market: 仅市场信号（de-vig 概率/赔率变动/离散度）
      - stats: 仅球队状态/排名（无市场、无裁判）
      - market_stats: 市场 + 球队状态（无裁判）
    """
    def g(key):
        v = feat_row.get(key)
        if v is None:
            return "NA"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    card = []
    if mode in ("full", "stats", "market_stats"):
        card.append("== HOME TEAM ==")
        card.append(f"  league rank: {g('H_rank')}   season points: {g('H_season_pts')}")
        card.append(f"  recent 5: GF avg={g('H_roll_GF')} GA avg={g('H_roll_GA')} "
                    f"pts avg={g('H_roll_Pts')} win rate={g('H_roll_Win')}")
        card.append(f"  shots avg={g('H_roll_Shots')} on-target={g('H_roll_ShotsT')} "
                    f"corners={g('H_roll_Corners')} fouls={g('H_roll_Fouls')}")
        card.append("== AWAY TEAM ==")
        card.append(f"  league rank: {g('A_rank')}   season points: {g('A_season_pts')}")
        card.append(f"  recent 5: GF avg={g('A_roll_GF')} GA avg={g('A_roll_GA')} "
                    f"pts avg={g('A_roll_Pts')} win rate={g('A_roll_Win')}")
        card.append(f"  shots avg={g('A_roll_Shots')} on-target={g('A_roll_ShotsT')} "
                    f"corners={g('A_roll_Corners')} fouls={g('A_roll_Fouls')}")
    if mode in ("full", "market", "market_stats"):
        card.append("== MARKET (pre-match) ==")
        card.append(f"  de-vig implied probs: H={g('mkt_prob_H')} D={g('mkt_prob_D')} A={g('mkt_prob_A')}")
        card.append(f"  odds move H={g('odds_move_H')} D={g('odds_move_D')} A={g('odds_move_A')} "
                    f"(positive = closing shorter than opening)")
        card.append(f"  closing-odds dispersion (bookmaker disagreement): {g('close_vol')}")
    if mode == "full":
        card.append("== REFEREE (historical, pre-match) ==")
        card.append(f"  avg cards={g('ref_cards_avg')} avg reds={g('ref_reds_avg')} "
                    f"avg fouls={g('ref_fouls_avg')} games officiated={g('ref_games')}")
    return "\n".join(card)


def build_messages(feat_row, mode="full"):
    """feat_row: dict（一行特征，赛前可得）。mode 见 build_feature_card。
    返回 messages 列表。"""
    card = build_feature_card(feat_row, mode=mode)
    user = (
        "Pre-match feature card for the match (information cutoff = kickoff time, "
        "no post-match information included):\n\n"
        f"{card}\n\n"
        "Produce your calibrated prediction as JSON."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
