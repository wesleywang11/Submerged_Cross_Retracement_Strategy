# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
from datetime import datetime

# ================= CONFIGURATION =================
# Tokyo Stock Exchange tickers
WATCH_LIST = [
    "6723.T", "9432.T", "7011.T", "7203.T", "8058.T", "8306.T", "9501.T", "285A.T",
    "6758.T", "9434.T", "2760.T", "9984.T", "8035.T", "9503.T", "4324.T", "9433.T",
    "7272.T", "6367.T", "6146.T", "6269.T", "6501.T", "8316.T", "5706.T", "5016.T",
    "7974.T", "7013.T", "4063.T", "4502.T", "6762.T", "6361.T", "6503.T", "8053.T",
    "7267.T", "6981.T", "6702.T", "8002.T", "4568.T", "9502.T", "1911.T", "5802.T"
]

LOOKBACK_BARS = 10  # Number of bars to look back

# =================================================

def detect_bullish_divergence_low(ticker):
    """
    Detect bullish divergence:
    - Current bar makes the lowest low within the last N bars
    - MACD histogram is rising

    Returns:
        (bool: divergence_found, dict: details)
    """
    try:
        # Download daily data
        df = yf.download(ticker, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)

        if df.empty or len(df) < 30:
            return False, {"error": "Insufficient data"}

        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']

        # MACD calculation
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        # Recent window
        recent_low = low.iloc[-(LOOKBACK_BARS + 1):]
        recent_hist = macd_hist.iloc[-(LOOKBACK_BARS + 1):]
        recent_close = close.iloc[-(LOOKBACK_BARS + 1):]
        recent_dates = df.index[-(LOOKBACK_BARS + 1):]

        current_low = float(recent_low.iloc[-1])
        current_hist = float(recent_hist.iloc[-1])
        current_close = float(recent_close.iloc[-1])
        current_date = recent_dates[-1].strftime('%Y-%m-%d')

        # Condition 1: lowest low in window
        min_low = float(recent_low.min())
        if current_low != min_low:
            return False, {"error": f"Not lowest low (min={min_low:.2f})"}

        # Condition 2: MACD histogram rising
        prev_hist_min = float(recent_hist.iloc[:-1].min())
        prev_hist_min_idx = recent_hist.iloc[:-1].idxmin()
        prev_hist_min_date = prev_hist_min_idx.strftime('%Y-%m-%d')

        if current_hist <= prev_hist_min:
            return False, {"error": "Histogram not rising"}

        current_dif = float(dif.iloc[-1])
        current_dea = float(dea.iloc[-1])

        hist_improvement = (
            (current_hist - prev_hist_min) / abs(prev_hist_min) * 100
            if prev_hist_min != 0 else 0
        )

        data = {
            "date": current_date,
            "price": current_close,
            "low": current_low,
            "hist": current_hist,
            "dif": current_dif,
            "dea": current_dea,
            "prev_hist_min": prev_hist_min,
            "prev_hist_date": prev_hist_min_date,
            "hist_improvement_pct": hist_improvement
        }

        return True, data

    except Exception as e:
        return False, {"error": str(e)}


def run_scanner():
    print("=" * 80)
    print(f" Daily MACD Bullish Divergence Scanner")
    print("=" * 80)
    print(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Watchlist size: {len(WATCH_LIST)}")
    print(f"Logic: Lowest low in last {LOOKBACK_BARS} bars + rising MACD histogram")
    print("-" * 80)

    results = []

    for ticker in WATCH_LIST:
        print(f"Analyzing {ticker:<8}...", end=" ")
        has_div, data = detect_bullish_divergence_low(ticker)

        if "error" in data:
            print(f"⚪ {data['error']}")
            continue

        if has_div:
            print("✅ Bullish divergence")
            results.append((ticker, data))
        else:
            print("⚪ No divergence")

    print("=" * 80)
    print(f"\nFound {len(results)} bullish divergence candidates\n")

    if results:
        results.sort(key=lambda x: x[1]['hist_improvement_pct'], reverse=True)

        print(f"{'Ticker':<8} {'Date':<12} {'Price':<10} {'Hist':<10} "
              f"{'PrevMinHist':<14} {'Improve%':<10} {'MACD'}")
        print("-" * 80)

        for ticker, d in results:
            macd_state = "Bullish GC" if d['dif'] > d['dea'] else "Bearish"
            print(f"{ticker:<8} {d['date']:<12} {d['price']:<10.2f} "
                  f"{d['hist']:<10.3f} {d['prev_hist_min']:<14.3f} "
                  f"+{d['hist_improvement_pct']:<9.1f}% {macd_state}")

    else:
        print("No candidates found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        run_scanner()
    except KeyboardInterrupt:
        print("\nProcess stopped by user")
    except Exception as e:
        print(f"\nProgram error: {e}")
