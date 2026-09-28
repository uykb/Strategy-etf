import urllib.request
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import re

def fetch_data():
    url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        df_btc = pd.DataFrame(json.loads(response.read().decode()), columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df_btc['date'] = pd.to_datetime(df_btc['open_time'], unit='ms').dt.date
    for col in ['open', 'high', 'low', 'close', 'volume']: df_btc[col] = df_btc[col].astype(float)
    df_btc['date'] = pd.to_datetime(df_btc['date'])
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get('https://farside.co.uk/bitcoin-etf-flow-all-data/', headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    parsed = [{'date_str': c[0].get_text(strip=True), 'etf_flow': -float(c[-1].get_text(strip=True)[1:-1]) if c[-1].get_text(strip=True).startswith('(') else float(c[-1].get_text(strip=True)) if c[-1].get_text(strip=True) not in ['-', '', 'NaN'] else 0.0} for r in soup.find_all('table')[0].find_all('tr') if len(c:=[col for col in r.find_all(['td', 'th'])])>=2 and re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', c[0].get_text(strip=True))]
    df_etf = pd.DataFrame(parsed)
    df_etf['date'] = pd.to_datetime(df_etf['date_str'], format='%d %b %Y', errors='coerce')
    
    df = pd.merge(df_btc, df_etf.dropna(subset=['date']), on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)
    
    df['etf_flow_ema3'] = df['etf_flow'].ewm(span=3, adjust=False).mean()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['sopr_28ma'] = (df['close'] / df['close'].ewm(span=28, adjust=False).mean()).rolling(28).mean()
    df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)
    
    for col in ['etf_flow_ema3', 'sopr_28ma', 'atr_14', 'vol_20']: df['signal_'+col] = df[col].shift(1)
    return df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')].copy().reset_index(drop=True)

df_bt = fetch_data()
vol_75th = df_bt['signal_vol_20'].quantile(0.75)

def run_sim(confirm_days=1, sopr_deadband=0.0, rebal_thresh=0.10):
    equity = 1000.0
    pos_weight = 0.0
    prev_target = 0.0
    peak_price, trough_price = None, None
    trade_count, stop_outs, total_fees = 0, 0, 0.0
    
    raw_target_history = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else row['open']
        atr_mult = 3.5 if row['signal_vol_20'] > vol_75th else 3.0
        stopped_out = False
        
        # Stop loss
        if pos_weight > 0 and peak_price is not None and row['low'] <= peak_price - atr_mult * row['signal_atr_14']:
            stopped_out, stop_outs = True, stop_outs + 1
            equity += equity * pos_weight * ((min(row['open'], peak_price - atr_mult * row['signal_atr_14']) - c_prev) / c_prev)
            fee = equity * abs(pos_weight) * 0.0010
            equity -= fee; total_fees += fee; pos_weight, peak_price = 0.0, None; trade_count += 1
        elif pos_weight < 0 and trough_price is not None and row['high'] >= trough_price + atr_mult * row['signal_atr_14']:
            stopped_out, stop_outs = True, stop_outs + 1
            equity += equity * abs(pos_weight) * ((c_prev - max(row['open'], trough_price + atr_mult * row['signal_atr_14'])) / c_prev)
            fee = equity * abs(pos_weight) * 0.0005
            equity -= fee; total_fees += fee; pos_weight, trough_price = 0.0, None; trade_count += 1

        if not stopped_out:
            s_flow, s_sopr = row['signal_etf_flow_ema3'], row['signal_sopr_28ma']
            
            # Base Target
            if s_flow > 10.0 and s_sopr > (1.0 + sopr_deadband):
                t_raw = 0.80 if s_flow > 150.0 and s_sopr > 1.02 else 0.50
            elif s_flow < -10.0 and s_sopr < (1.0 - sopr_deadband):
                t_raw = -0.50 if s_flow < -100.0 else -0.35
            elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
                t_raw = 0.10
            else:
                t_raw = prev_target

            raw_target_history.append(t_raw)
            
            # Confirmation logic
            if len(raw_target_history) >= confirm_days:
                recent_targets = raw_target_history[-confirm_days:]
                if all(x == recent_targets[0] for x in recent_targets): confirmed_target = recent_targets[0]
                else: confirmed_target = prev_target
            else:
                confirmed_target = prev_target
            
            prev_target = confirmed_target
            
            # Volatility cap
            target_weight = min(confirmed_target, 0.40) if row['signal_vol_20'] > vol_75th and confirmed_target > 0 else (max(confirmed_target, -0.40) if row['signal_vol_20'] > vol_75th and confirmed_target < 0 else confirmed_target)

            # Rebalance
            if abs(target_weight - pos_weight) >= rebal_thresh:
                fee = equity * abs(target_weight - pos_weight) * (0.0010 if target_weight >= 0 else 0.0005)
                equity -= fee; total_fees += fee; trade_count += 1; pos_weight = target_weight
                peak_price = row['high'] if pos_weight > 0 else None
                trough_price = row['low'] if pos_weight < 0 else None
            else:
                if pos_weight > 0 and peak_price is not None: peak_price = max(peak_price, row['high'])
                elif pos_weight < 0 and trough_price is not None: trough_price = min(trough_price, row['low'])

            # PnL
            if pos_weight > 0: equity += equity * pos_weight * ((row['close'] - c_prev) / c_prev)
            elif pos_weight < 0: equity += equity * abs(pos_weight) * ((c_prev - row['close']) / c_prev) + equity * abs(pos_weight) * 0.0001
            
    return equity, trade_count, total_fees

print(f"1. Base Opt (1-day, 0 deadband, 10% thr): {run_sim(1, 0.0, 0.10)}")
print(f"2. 2-Day Confirm (2-day, 0 deadband, 10% thr): {run_sim(2, 0.0, 0.10)}")
print(f"3. 3-Day Confirm (3-day, 0 deadband, 10% thr): {run_sim(3, 0.0, 0.10)}")
print(f"4. Deadband + 15% (1-day, 0.01 deadband, 15% thr): {run_sim(1, 0.01, 0.15)}")
print(f"5. Deadband + 2-Day (2-day, 0.01 deadband, 15% thr): {run_sim(2, 0.01, 0.15)}")
