"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Model: per-trophy logit trend against log time, with shrinkage, a
dilution-fade damp, a stale-page refresh adjustment, and a calibrated
spread. Standard library only, no network, fully deterministic.

Why this shape
--------------
An earn rate is holders / players-tracked. Both move: tracked players
grow as more profiles sync (diluting every rate), while holders grow as
players reach each trophy (lifting it). On launch night the page sat
unchanged for hours (638 players tracked), then one refresh jumped it
to 3,868 players and every rate but the 100% one fell: a refresh
dilutes. Over a week the two forces trade places -- early-story
trophies climb fast as everyone finishes chapter 1, late trophies
stay rare while the tracked base keeps growing -- so each trophy gets
its own trend, and the trend is fitted on the logit scale against
ln(1 + hours since the release day began), which damps extrapolation
the way a launch curve damps: fast early, slow late.

What it does per trophy
------------------------
1. Collapse the history to distinct values (Exophase refreshes in
   steps, so consecutive identical reads carry no new information).
2. Weighted least squares of logit(rate) on ln(1 + t), weights halving
   per day of age, so the recent slope leads.
3. Shrink the slope toward its tier's mean slope (platinum/gold/
   silver/bronze behave alike), more shrinkage with fewer points.
4. Damp a negative slope by a quarter when holders are still growing:
   a falling rate with growing holders is dilution, and dilution fades
   as the tracked base saturates, while progress does not.
5. Cap the total logit move from the last read at 2.5, so no
   extrapolation runs away over the 4.5-day final horizon.
6. When the history is flat (no rate changed at all, as on launch
   night), a refresh is due: the forecast becomes a two-part mixture
   -- with the chance a refresh lands before the target the rate
   drops ~1.8 logit (the launch-night refresh diluted tracked
   players 638 -> 3,868, a 6.06x jump, and ln(6.06) is 1.80),
   otherwise it stays. The mixture's quantiles have a long lower
   tail instead of a symmetric band, which is what a stale page
   honestly implies. The 100% trophy is exempt: every new player
   holds it, so a refresh does not dilute it.
7. Spread (logit sigma, trending histories only) combines the
   random-walk term, the fit residual, and the extrapolation
   distance times the slope, floored so the forecast is never
   overconfident.

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The release day begins here; log time is measured from it.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

SIGMA_RW = 0.5          # random-walk spread, logit per sqrt(day)
MAX_LOGIT_MOVE = 2.5    # cap on the median's logit move from the last read
REFRESH_DROP = 1.8      # logit a stale-page refresh costs a rate: the launch-night
                      # refresh took tracked players 638 -> 3,868, a 6.06x
                      # dilution, and ln(6.06) is 1.80
REFRESH_CYCLE_H = 4.5   # prior for hours between Exophase refreshes
SHRINK_K = 2.0          # pseudo-count pulling a slope to its tier mean
NEG_DAMP = 0.75         # negative slopes damp this much when holders grow
MIX_S_REFRESH = 0.3     # within-refresh spread of the refresh drop, logit
MIX_S_STAY = 0.1        # spread when no refresh lands, logit


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(lg_last, tau, p_refresh, drop=REFRESH_DROP):
    """Quantile of a two-part belief about a flat page's next rate.

    With probability p_refresh a refresh lands before the target and
    the rate drops by `drop` logit (spread MIX_S_REFRESH); otherwise
    it stays where it is (spread MIX_S_STAY). Inverting the mixture
    CDF by bisection. A flat page is a stale page, and a refresh
    dilutes: new tracked players hold far fewer trophies than the
    early cohort, so the honest forecast has a long lower tail, not a
    symmetric band around the last rate.
    """
    if p_refresh <= 0.0:
        # No refresh in play: a narrow band around the last rate.
        z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}[tau]
        return lg_last + z * MIX_S_STAY
    lo, hi = lg_last - 6.0, lg_last + 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = p_refresh * norm_cdf((mid - (lg_last - drop)) / MIX_S_REFRESH) \
            + (1.0 - p_refresh) * norm_cdf((mid - lg_last) / MIX_S_STAY)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        resid = math.sqrt(
            sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
            / (sw * (n - 2) / n)
        )
    else:
        resid = 0.0
    return (a, b, resid)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    checkpoint = parse_time(case["at"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    # Target time is checkpoint + horizon; use its log time directly.
    x_target = math.log1p(hours_since_epoch(checkpoint) + horizon_h)

    # Per-trophy distinct-value series: (log-time x, logit rate, hours).
    series = [[] for _ in range(n_tr)]
    last_rates = [None] * n_tr
    # Global change times: when any trophy's rate moved (refresh steps).
    change_times = []
    prev_rates = None
    first_at = parse_time(history[0]["at"])
    for read in history:
        t = hours_since_epoch(parse_time(read["at"]))
        x = math.log1p(max(0.0, t))
        rates = read["rates"]
        if prev_rates is not None and rates != prev_rates:
            change_times.append(t)
        prev_rates = rates
        for i in range(n_tr):
            r = rates[i]
            if r is None:
                continue
            if last_rates[i] is None or r != last_rates[i][0]:
                series[i].append((x, logit(r), t))
                last_rates[i] = (r, t)
    last_read = history[-1]
    last_rates = last_read["rates"]
    last_t = hours_since_epoch(parse_time(last_read["at"]))
    x_last = math.log1p(max(0.0, last_t))

    # Tracked-players series, for the holders-growth check.
    n_first = history[0]["players_tracked"]
    n_last = last_read["players_tracked"]

    # First pass: per-trophy fits, and tier-mean slopes.
    fits = []
    tier_slopes = {}
    for i in range(n_tr):
        pts = series[i]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [0.5 ** ((last_t - p[2]) / 24.0) for p in pts]
            fit = wls(xs, ys, ws)
        else:
            fit = None
        fits.append(fit)
        if fit is not None and len(pts) >= 3:
            tier_slopes.setdefault(trophies[i]["tier"], []).append(fit[1])
    tier_mean = {k: sum(v) / len(v) for k, v in tier_slopes.items()}

    # Tier-median residual, for trophies with too few points to measure one.
    resids = [f[2] for f in fits if f is not None and f[2] > 0]
    resids.sort()
    median_resid = resids[len(resids) // 2] if resids else 0.3

    # Is the page flat / stale? No rate changed anywhere in the history.
    flat = len(change_times) == 0
    if flat:
        stale_h = (checkpoint - first_at).total_seconds() / 3600.0
        # A refresh is overdue once a full cycle has elapsed: the chance
        # one lands inside the horizon, capped so it is never certain.
        p_refresh = min(0.70, max(0.15, (stale_h + horizon_h) / REFRESH_CYCLE_H))
    else:
        gaps = []
        prev = None
        for t in change_times:
            if prev is not None:
                gaps.append(t - prev)
            prev = t
        gaps.sort()
        interval = gaps[len(gaps) // 2] if gaps else REFRESH_CYCLE_H
        interval = max(1.0, min(12.0, interval))
        p_refresh = 1.0 - math.exp(-horizon_h / interval)

    rw = SIGMA_RW * math.sqrt(horizon_d)
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            # Fall back to the last non-null logit for this trophy.
            lg_last = series[i][-1][1] if series[i] else logit(0.01)
        else:
            lg_last = logit(r_last)
        fit = fits[i]
        n_pts = len(series[i])
        if fit is not None:
            a, b, resid = fit
            b_tier = tier_mean.get(trophy["tier"], 0.0)
            b = (n_pts * b + SHRINK_K * b_tier) / (n_pts + SHRINK_K)
            # Dilution fade: a falling rate with growing holders is
            # dilution, and dilution fades as the tracked base saturates.
            if b < 0 and n_last > n_first:
                holders_first = None
                holders_last = None
                for read in history:
                    rate = read["rates"][i]
                    if rate is None:
                        continue
                    holders = rate * read["players_tracked"] / 100.0
                    if holders_first is None:
                        holders_first = holders
                    holders_last = holders
                if holders_last is not None and holders_first is not None \
                        and holders_last > holders_first:
                    b *= NEG_DAMP
            lg_med = a + b * x_target
            if resid <= 0:
                resid = median_resid
        else:
            lg_med = lg_last
            b = 0.0
            resid = median_resid
        if flat:
            # Stale page: the two-part refresh mixture, not a trend.
            # The 100% trophy is exempt -- a refresh does not dilute it.
            p = 0.0 if (r_last is not None and r_last >= 99.99) else p_refresh
            qs = []
            prev_q = 0.01
            for tau in LEVELS:
                q = round4(clamp(inv_logit(mixture_quantile(lg_last, tau, p))))
                if q < prev_q:
                    q = prev_q
                if q > 99.99:
                    q = 99.99
                qs.append(q)
                prev_q = q
            rows.append({"id": trophy["id"], "q": qs})
            continue
        # Cap the move from the last read.
        lg_med = min(lg_last + MAX_LOGIT_MOVE, max(lg_last - MAX_LOGIT_MOVE, lg_med))
        # Spread: random walk + fit residual + extrapolation.
        trend_unc = 0.5 * abs(b) * max(0.0, x_target - x_last)
        sigma = math.sqrt(rw * rw + resid * resid + trend_unc * trend_unc)
        sigma = max(0.5, min(2.0, sigma))
        qs = []
        prev_q = 0.01
        for z in Z:
            q = round4(clamp(inv_logit(lg_med + z * sigma)))
            if q < prev_q:
                q = prev_q
            if q > 99.99:
                q = 99.99
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
