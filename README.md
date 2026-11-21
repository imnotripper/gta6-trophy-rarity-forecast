# GTA VI Trophy Rarity Forecast

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast, for
every PlayStation 5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will have earned it at the case's target time, as
five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Scored by mean pinball loss
on the logit scale.

## Run

```
python3 main.py            # reads $MAKEFUN_INPUT_DIR (else /input),
                           # writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Python 3.13, standard library only, no network, deterministic.

## Model

All modelling is on the logit scale, where the score lives:

1. **Snapshot compression** — Exophase refreshes its counts in steps, so
   consecutive identical reads collapse into one snapshot.
2. **Trajectory fit** — per trophy, a recency-weighted least-squares line of
   `logit(rate)` against `ln(1 + hours since release)` over the last ~48 hours
   of snapshots. Earn rates grow roughly logistically, so this is the natural
   first-order curve; it is read at the target time.
3. **Tier shrinkage** — each trophy's slope is shrunk toward the median slope
   of its tier (platinum / gold / silver / bronze). Story trophies run fast
   and high; completionist and collectible trophies run slow and low; a noisy
   rare trophy borrows strength from its class instead of extrapolating its
   own jitter.
4. **Dilution blend** — `players_tracked` and the trophy's earner count
   (`rate × players`) are also fit separately on the log scale and divided at
   the target. When a surge of newly tracked players dilutes a rate, this
   term pulls the forecast down even while earners keep growing; it is
   blended in at 20% when the earner count is solid.
5. **Honest spread** — the quantile width combines a random-walk term growing
   with `sqrt(horizon)`, the fit's residual scatter, and the extrapolation
   distance. Flat or tiny histories (e.g. a page Exophase has not refreshed
   yet) fall back to the last rate with a wide, refresh-aware spread, and
   tier priors cover trophies with no usable rate at all.
