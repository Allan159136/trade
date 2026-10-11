"""Récupère les bougies, calcule les indicateurs et écrit un JSON par marché dans data/.
Lancé chaque heure par GitHub Actions (voir update.yml)."""
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

API_KEY = os.environ["TWELVE_DATA_KEY"]  # stockée dans les "Secrets" GitHub
OUT = "data"

# nom affiché -> symbole chez Twelve Data (vérifie que ton plan couvre le Brent)
MARKETS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPJPY": "GBP/JPY",
    "BRENT": "BRENT",
}
# timeframe affiché -> intervalle Twelve Data
TFS = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"}

PAUSE = 8  # secondes entre deux appels (le plan gratuit limite à ~8 appels/minute)

# Marchés ouverts 7j/7 (crypto...) : on ne filtre pas le week-end pour eux. Ex. {"BTCUSD"}
ALWAYS_OPEN = set()


def is_closed(ts) -> bool:
    """Or et forex : fermés du vendredi 21h UTC au dimanche 22h UTC.
    Le week-end, Twelve Data renvoie des bougies "bouche-trou" (presque plates) qu'on retire."""
    wd, h = ts.weekday(), ts.hour  # lundi = 0 ... dimanche = 6
    return wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22)


def fetch(symbol: str, interval: str, weekend_filter: bool = True) -> pd.DataFrame:
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": interval, "outputsize": 600, "timezone": "UTC", "apikey": API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(j.get("message", "réponse inattendue"))
    df = pd.DataFrame(j["values"]).iloc[::-1].reset_index(drop=True)  # du plus ancien au plus récent
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)
    if weekend_filter:
        df = df[~pd.to_datetime(df["datetime"]).map(is_closed)].reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError(f"pas assez de bougies réelles ({len(df)})")
    return df


def snapshot(df: pd.DataFrame) -> dict:
    close = df["close"]
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    hist = macd - macd.ewm(span=9).mean()

    score = 0
    score += 1 if close.iloc[-1] > ema50.iloc[-1] else -1
    score += 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
    score += 1 if hist.iloc[-1] > 0 else -1
    score += 1 if rsi.iloc[-1] > 55 else (-1 if rsi.iloc[-1] < 45 else 0)
    bias = "haussier" if score >= 2 else "baissier" if score <= -2 else "neutre"

    last24 = df.tail(24)
    return {
        "ts": df["datetime"].iloc[-1],
        "close": round(close.iloc[-1], 5),
        "chg_1": round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 3),
        "chg_24": round((close.iloc[-1] / close.iloc[-25] - 1) * 100, 3),
        "high_24": round(last24["high"].max(), 5),
        "low_24": round(last24["low"].min(), 5),
        "ema50": round(ema50.iloc[-1], 5),
        "rsi": round(rsi.iloc[-1], 1),
        "macd_hist": round(hist.iloc[-1], 5),
        "score": int(score),
        "bias": bias,
    }


def summary(name: str, s: dict) -> str:
    trend = "au-dessus de" if s["close"] > s["ema50"] else "sous"
    mom = "positif" if s["macd_hist"] > 0 else "négatif"
    if s["rsi"] > 70:
        r = "en zone de surachat, attention à un possible repli"
    elif s["rsi"] < 30:
        r = "en zone de survente, un rebond est possible"
    elif s["rsi"] > 55:
        r = "plutôt haut sans être en surchauffe"
    elif s["rsi"] < 45:
        r = "plutôt bas, la pression vendeuse domine"
    else:
        r = "neutre"
    head = {
        "haussier": "Le biais est haussier",
        "baissier": "Le biais est baissier",
        "neutre": "Le biais est neutre, le marché hésite",
    }[s["bias"]]
    return (f"{head} sur {name} : le prix évolue {trend} sa moyenne mobile 50, "
            f"le momentum (MACD) est {mom} et le RSI est {r} ({s['rsi']:.0f}).")


def candles(df: pd.DataFrame, n: int = 40) -> list:
    return [[row.datetime, row.open, row.high, row.low, row.close] for row in df.tail(n).itertuples()]


def history(df: pd.DataFrame, n: int = 12) -> list:
    rows = []
    for i in range(n):
        end = len(df) - i
        s = snapshot(df.iloc[:end])
        rows.append({"ts": s["ts"], "close": s["close"], "chg_1": s["chg_1"], "rsi": s["rsi"], "bias": s["bias"]})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    done = []
    for name, symbol in MARKETS.items():
        closed = name not in ALWAYS_OPEN and is_closed(pd.Timestamp.now(tz="UTC"))
        data = {"market": name, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "market_closed": bool(closed), "tf": {}}
        try:
            for tf, interval in TFS.items():
                df = fetch(symbol, interval, name not in ALWAYS_OPEN)
                snap = snapshot(df)
                data["tf"][tf] = {"snapshot": snap, "summary": summary(name, snap), "candles": candles(df)}
                if tf == "H1":
                    data["history"] = history(df)
                time.sleep(PAUSE)
        except Exception as e:  # en cas d'erreur, on garde l'ancien fichier
            print(f"[{name}] erreur : {e}")
            continue
        with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        done.append(name)
        print(f"[{name}] ok")

    with open(f"{OUT}/index.json", "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "markets": done}, f)


if __name__ == "__main__":
    main()
