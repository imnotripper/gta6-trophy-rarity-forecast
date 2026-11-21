"""gta6-trophy-rarity-forecast-01 entry: forecast GTA VI trophy earn rates.

Reads each case file (<id>.in) in the input directory and writes <id>.out with,
per trophy, five quantiles (0.05/0.25/0.5/0.75/0.95) of its Exophase earn rate
(percent of tracked players) at the case's target time. Scoring is mean pinball
loss on the logit scale, so all modelling happens in logit space.

Model, per trophy:
  1. Compress the read history to distinct page snapshots (Exophase refreshes
     its counts in steps, so consecutive reads are often identical).
  2. Fit a weighted least-squares line of logit(rate) against ln(1 + hours
     since release) over the recent snapshots (recency-weighted, 30 h
     half-life). Earn rates grow roughly logistically, so this line is the
     natural first-order trajectory.
  3. Shrink each trophy's slope toward the median slope of its tier
     (platinum/gold/silver/bronze): story trophies run fast and high,
     completionist trophies slow and low, and a noisy rare trophy borrows
     strength from its class.
  4. Blend in a decomposition forecast: fit players_tracked and the trophy's
     earner count (rate x players) separately on the log scale and divide.
     This captures dilution, when a surge of newly tracked players pulls a
     rate down even while earners keep growing.
  5. Emit quantiles around the blended centre with a spread that grows with
     the horizon (random-walk term), the fit's residual scatter, and the
     extrapolation distance. Flat or tiny histories fall back to the last
     rate with a wide, refresh-aware spread.

Deterministic, standard library only, no network. Runs in well under a second
for a 55-trophy case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS_FALLBACK = [0.05, 0.25, 0.5, 0.75, 0.95]
NORM = NormalDist()

# Fallback centre (percent) for a trophy with no usable rate anywhere, by tier.
TIER_PRIOR_PCT = {"platinum": 0.4, "gold": 1.5, "silver": 4.0, "bronze": 8.0}
# Generous week-one ceilings (percent) by tier; they only bind on wild fits.
TIER_CAP_PCT = {"platinum": 50.0, "gold": 90.0, "silver": 97.0, "bronze": 99.9}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(anchor, moment):
    return (moment - anchor).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns dict or None."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-9:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    return {"a": a, "b": b, "xbar": xbar, "ybar": ybar, "sxx": sxx,
            "resid_sd": math.sqrt(max(var, 0.0)), "weight": sw, "n": n}


def distinct_snapshots(history):
    """Drop consecutive reads whose page content is identical."""
    snaps = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(
            None if r is None else round(float(r), 2)
            for r in read.get("rates", [])))
        if key != prev_key:
            snaps.append(read)
            prev_key = key
    return snaps


def last_known_rate(snaps, idx):
    for read in reversed(snaps):
        rates = read.get("rates", [])
        if idx < len(rates) and rates[idx] is not None:
            return float(rates[idx])
    return None


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS_FALLBACK
    zs = [NORM.inv_cdf(min(0.999, max(0.001, float(tau)))) for tau in levels]

    anchor = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else None
    if anchor is None:
        anchor = parse_time(history[0]["at"]) if history else \
            datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case["target"])
    horizon_days = max(float(case.get("horizon_minutes", 0)) / 1440.0, 0.0)
    x_target = math.log1p(max(hours_since(anchor, target), 0.0))

    snaps = distinct_snapshots(history)
    snap_t = []
    for read in snaps:
        moment = parse_time(read.get("slot") or read["at"])
        snap_t.append(hours_since(anchor, moment))
    t_last = snap_t[-1] if snap_t else 0.0

    # Fit window: the last 48 hours of snapshots, widening if too few remain.
    win = [i for i, t in enumerate(snap_t) if t >= t_last - 48.0]
    if len(win) < 4:
        win = list(range(len(snaps)))
    w_recency = [max(0.05, 0.5 ** ((t_last - snap_t[i]) / 30.0)) for i in win]
    xs_win = [math.log1p(max(snap_t[i], 0.0)) for i in win]

    # Players-tracked trajectory (for the decomposition blend).
    n_fit = None
    if len(win) >= 3:
        nxs, nys = [], []
        for pos, i in enumerate(win):
            players = snaps[i].get("players_tracked")
            if players:
                nxs.append(xs_win[pos])
                nys.append(math.log(float(players)))
        if len(nxs) >= 3:
            n_fit = wls(nxs, nys, [1.0] * len(nxs))
    n_last = None
    for read in reversed(snaps):
        if read.get("players_tracked"):
            n_last = float(read["players_tracked"])
            break
    n_target = None
    if n_fit is not None and n_last:
        ln_n = min(20.0, n_fit["a"] + n_fit["b"] * x_target)
        n_target = min(max(math.exp(ln_n), n_last * 0.9), n_last * 10.0)

    # Pass 1: raw per-trophy logit/log-time fits.
    raw = []
    for idx in range(len(trophies)):
        xs, ys, ws = [], [], []
        for pos, i in enumerate(win):
            rates = snaps[i].get("rates", [])
            if idx < len(rates) and rates[idx] is not None:
                xs.append(xs_win[pos])
                ys.append(logit(float(rates[idx])))
                ws.append(w_recency[pos])
        raw.append(wls(xs, ys, ws) if len(ys) >= 3 else None)

    slopes = [f["b"] for f in raw if f is not None]
    global_slope = median(slopes) if slopes else 0.8
    tier_slope = {}
    for tier in TIER_PRIOR_PCT:
        ts = [raw[i]["b"] for i, tr in enumerate(trophies)
              if tr.get("tier") == tier and raw[i] is not None]
        tier_slope[tier] = median(ts) if ts else global_slope

    rows = []
    sqrt_h = math.sqrt(horizon_days)
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        r_last = last_known_rate(snaps, idx)
        fit = raw[idx]
        if fit is not None and r_last is not None:
            prior = tier_slope.get(tier, global_slope)
            b = (fit["weight"] * fit["b"] + 3.0 * prior) / (fit["weight"] + 3.0)
            b = min(7.0, max(-2.0, b))
            x_last = math.log1p(max(t_last, 0.0))
            y_last = logit(r_last)
            mu_line = fit["ybar"] + b * (x_target - fit["xbar"])
            mu_anchor = y_last + b * (x_target - x_last)
            mu = 0.55 * mu_line + 0.45 * mu_anchor
            # Decomposition blend: earners vs players, when both are solid.
            if n_target and n_last and r_last is not None:
                earners_last = r_last / 100.0 * n_last
                if earners_last >= 5.0 and len(win) >= 3:
                    exs, eys = [], []
                    for pos, i in enumerate(win):
                        rates = snaps[i].get("rates", [])
                        players = snaps[i].get("players_tracked")
                        if players and idx < len(rates) and rates[idx] is not None:
                            exs.append(xs_win[pos])
                            eys.append(math.log(float(rates[idx]) / 100.0
                                               * float(players) + 0.5))
                    e_fit = wls(exs, eys, [1.0] * len(exs)) if len(exs) >= 3 else None
                    if e_fit is not None:
                        ln_e = min(20.0, e_fit["a"] + e_fit["b"] * x_target)
                        e_target = math.exp(ln_e)
                        r_decomp = clamp_pct(100.0 * e_target / n_target)
                        mu = 0.8 * mu + 0.2 * logit(r_decomp)
            extrap = min(2.0, max(0.0, x_target - x_last))
            sigma = math.sqrt((0.45 * sqrt_h) ** 2
                              + (0.55 * fit["resid_sd"]) ** 2
                              + (0.18 * extrap) ** 2 + 0.12 ** 2)
            sigma = min(1.85, max(0.28, sigma))
        else:
            centre = r_last if r_last is not None \
                else TIER_PRIOR_PCT.get(tier, 8.0)
            mu = logit(centre)
            sigma = 0.55 * sqrt_h + 0.35
            if r_last is None:
                sigma = max(sigma, 1.1)
        cap = TIER_CAP_PCT.get(tier, 99.9)
        qs = []
        for z in zs:
            pct = inv_logit(mu + z * sigma)
            qs.append(round4(min(cap, max(0.0, pct))))
        # Rounding must never break the non-decreasing order.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
