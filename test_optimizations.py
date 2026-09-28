import urllib.request
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import re

def fetch_btc_data():
    url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
    return df

def fetch_farside_etf_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get('https://farside.co.uk/bitcoin-etf-flow-all-data/', headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    parsed = []
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')
    for r in rows:
        cols = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
        if len(cols) >= 2:
            date_str = cols[0]
            total_str = cols[-1]
            if re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', date_str):
                val_str = total_str.replace('$', '').replace(',', '').strip()
                if val_str.startswith('(') and val_str.endswith(')'):
                    val = -float(val_str[1:-1])
                elif val_str in ['-', '', 'NaN', '(0.0)']:
                    val = 0.0
                else:
                    try:
                        val = float(val_str)
                    except:
                        val = 0.0
                parsed.append({'date_str': date_str, 'etf_flow': val})
    
    df_etf = pd.DataFrame(parsed)
    df_etf['date'] = pd.to_datetime(df_etf['date_str'], format='%d %b %Y', errors='coerce')
    df_etf = df_etf.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df_etf = df_etf[['date', 'etf_flow']].drop_duplicates(subset=['date'])
    return df_etf

df_btc = fetch_btc_data()
df_etf = fetch_farside_etf_data()
df = pd.merge(df_btc, df_etf, on='date', how='left')
df['etf_flow'] = df['etf_flow'].fillna(0.0)

# Indicators
prev_close = df['close'].shift(1)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - prev_close).abs()
tr3 = (df['low'] - prev_close).abs()
df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
df['atr_14'] = df['tr'].rolling(14).mean()

cost_basis = df['close'].ewm(span=28, adjust=False).mean()
df['sopr'] = df['close'] / cost_basis
df['sopr_28ma'] = df['sopr'].rolling(28).mean()

df['ret'] = df['close'].pct_change()
df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(365)

# Smooth ETF flow with 3-day EMA & 5-day SMA
df['etf_flow_ema3'] = df['etf_flow'].ewm(span=3, adjust=False).mean()
df['etf_flow_sma5'] = df['etf_flow'].rolling(5).mean()

# Shifts
df['signal_flow_raw'] = df['etf_flow'].shift(1)
df['signal_flow_ema3'] = df['etf_flow_ema3'].shift(1)
df['signal_flow_sma5'] = df['etf_flow_sma5'].shift(1)
df['signal_sopr'] = df['sopr_28ma'].shift(1)
df['signal_atr'] = df['atr_14'].shift(1)
df['signal_vol20'] = df['vol_20'].shift(1)

bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol20'].quantile(0.75)

def run_sim(flow_type='raw', threshold=0.0, atr_mult=3.0, rebalance_min_delta=0.0):
    initial_capital = 1000.0
    equity = initial_capital
    pos_weight = 0.0
    prev_target_raw = 0.0
    peak_price = None
    trough_price = None
    
    trade_count = 0
    total_funding = 0.0
    total_fees = 0.0
    stop_outs = 0
    equity_curve = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        dt = row['date']
        op = row['open']
        hi = row['high']
        lo = row['low']
        cl = row['close']
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
        
        if flow_type == 'raw':
            s_flow = row['signal_flow_raw']
        elif flow_type == 'ema3':
            s_flow = row['signal_flow_ema3']
        elif flow_type == 'sma5':
            s_flow = row['signal_flow_sma5']
            
        s_sopr = row['signal_sopr']
        s_atr = row['signal_atr']
        s_vol = row['signal_vol20']
        
        stopped_out = False
        
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - atr_mult * s_atr
            if lo <= stop_price:
                stopped_out = True
                stop_outs += 1
                exec_price = min(op, stop_price)
                day_ret = (exec_price - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee
                total_fees += fee
                pos_weight, peak_price = 0.0, None
                trade_count += 1
                
        elif pos_weight < 0 and trough_price is not None:
            stop_price = trough_price + atr_mult * s_atr
            if hi >= stop_price:
                stopped_out = True
                stop_outs += 1
                exec_price = max(op, stop_price)
                day_ret = (c_prev - exec_price) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee
                total_fees += fee
                pos_weight, trough_price = 0.0, None
                trade_count += 1

        if not stopped_out:
            if s_flow > threshold and s_sopr > 1.0:
                target_raw = 0.75 # Long 75%
            elif s_flow < -threshold and s_sopr < 1.0:
                target_raw = -0.45 # Short 45%
            elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
                target_raw = 0.10 # Contradiction 10%
            else:
                target_raw = prev_target_raw
                
            prev_target_raw = target_raw
            
            if s_vol > vol_75th:
                target_weight = min(target_raw, 0.40) if target_raw > 0 else (max(target_raw, -0.40) if target_raw < 0 else 0.0)
            else:
                target_weight = target_raw
                
            # Rebalance threshold filter to avoid small fee attrition
            if abs(target_weight - pos_weight) >= rebalance_min_delta:
                turnover = abs(target_weight - pos_weight)
                fee_rate = 0.0010 if target_weight >= 0 else 0.0005
                fee = equity * turnover * fee_rate
                equity -= fee
                total_fees += fee
                trade_count += 1
                pos_weight = target_weight
                
                if pos_weight > 0:
                    peak_price, trough_price = hi, None
                elif pos_weight < 0:
                    trough_price, peak_price = lo, None
                else:
                    peak_price, trough_price = None, None
            else:
                if pos_weight > 0 and peak_price is not None:
                    peak_price = max(peak_price, hi)
                elif pos_weight < 0 and trough_price is not None:
                    trough_price = min(trough_price, lo)

            if pos_weight > 0:
                equity += equity * pos_weight * ((cl - c_prev) / c_prev)
            elif pos_weight < 0:
                equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev)
                funding_earned = equity * abs(pos_weight) * 0.0001
                equity += funding_earned
                total_funding += funding_earned

        equity_curve.append(equity)
        
    df_eval = pd.DataFrame({'equity': equity_curve})
    ret = (equity - initial_capital) / initial_capital
    ann_ret = (1 + ret) ** (365.0 / len(df_bt)) - 1
    df_eval['peak'] = df_eval['equity'].cummax()
    mdd = ((df_eval['equity'] - df_eval['peak']) / df_eval['peak']).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    daily_ret = df_eval['equity'].pct_change().fillna(0.0)
    sharpe = (daily_ret.mean() - 0.02/365) / daily_ret.std() * np.sqrt(365) if daily_ret.std() > 0 else 0
    
    return {
        'final_equity': equity,
        'ret': ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'calmar': calmar,
        'sharpe': sharpe,
        'trades': trade_count,
        'stop_outs': stop_outs,
        'fees': total_fees
    }

res_base = run_sim(flow_type='raw', threshold=0.0, atr_mult=3.0, rebalance_min_delta=0.0)
res_ema3 = run_sim(flow_type='ema3', threshold=10.0, atr_mult=3.5, rebalance_min_delta=0.10)
res_sma5 = run_sim(flow_type='sma5', threshold=20.0, atr_mult=3.5, rebalance_min_delta=0.10)

print("=== OPTIMIZATION VARIANT COMPARISON ===")
print(f"Base Strategy: Final=${res_base['final_equity']:.2f}, Ret={res_base['ret']*100:.2f}%, MDD={res_base['mdd']*100:.2f}%, Sharpe={res_base['sharpe']:.2f}, Trades={res_base['trades']}, Fees=${res_base['fees']:.2f}")
print(f"EMA3+Filter Strategy: Final=${res_ema3['final_equity']:.2f}, Ret={res_ema3['ret']*100:.2f}%, MDD={res_ema3['mdd']*100:.2f}%, Sharpe={res_ema3['sharpe']:.2f}, Trades={res_ema3['trades']}, Fees=${res_ema3['fees']:.2f}")
print(f"SMA5+Filter Strategy: Final=${res_sma5['final_equity']:.2f}, Ret={res_sma5['ret']*100:.2f}%, MDD={res_sma5['mdd']*100:.2f}%, Sharpe={res_sma5['sharpe']:.2f}, Trades={res_sma5['trades']}, Fees=${res_sma5['fees']:.2f}")
