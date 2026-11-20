# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/scriptstealer/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/scriptstealer/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by pinball loss on the logit scale.

How it works: an earn rate is holders divided by players tracked, and both move. Tracked players grow as more profiles sync, diluting every rate; holders grow as players reach each trophy, lifting it. On launch night the page sat unchanged for hours at 638 players, then one refresh jumped it to 3,868 and every rate but the 100% one fell. So each trophy gets its own logit trend against ln(1 + hours since the release day began), fitted on distinct values only (Exophase refreshes in steps), shrunk toward its tier's mean slope, with a falling slope damped when holders are still growing (dilution fades, progress does not), the total move capped, and a refresh adjustment when the page is flat and stale: a refresh is due, so the median shifts down and the spread widens. Spread combines a random-walk term, the fit residual, extrapolation distance, and refresh uncertainty. Standard library only, no network, fully deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6rdgs9pm&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
