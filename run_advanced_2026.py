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
    parsed = []
    for r in soup.find_all('table')[0].find_all('tr'):
        cols = [col.get_text(strip=True) for col in r.find_all(['td', 'th'])]
        if len(cols) >= 2 and re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', cols[0]):
            v_str = cols[-1].replace('$', '').replace(',', '').strip()
            v = -float(v_str[1:-1]) if v_str.startswith('(') else float(v_str) if v_str not in ['-', '', 'NaN'] else 0.0
            parsed.append({'date_str': cols[0], 'etf_flow': v})
    df_etf = pd.DataFrame(parsed)
    df_etf['date'] = pd.to_datetime(df_etf['date_str'], format='%d %b %Y', errors='coerce')
    
    df = pd.merge(df_btc, df_etf.dropna(subset=['date']), on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)
    
    # Calculate Indicators
    df['etf_flow_ema3'] = df['etf_flow'].ewm(span=3, adjust=False).mean()
    prev_c = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_c).abs(), (df['low'] - prev_c).abs()))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['sopr_28ma'] = (df['close'] / df['close'].ewm(span=28, adjust=False).mean()).rolling(28).mean()
    df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Shift to prevent lookahead
    for col in ['etf_flow_ema3', 'sopr_28ma', 'atr_14', 'vol_20', 'sma_50']:
        df['signal_'+col] = df[col].shift(1)
        
    return df

df = fetch_data()
bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol_20'].quantile(0.75)

equity = 1000.0
pos_weight = 0.0
prev_confirmed_target = 0.0
peak_price, trough_price = None, None
trade_count, stop_outs, total_fees = 0, 0, 0.0
raw_target_history = []
equity_curve = []

for i in range(len(df_bt)):
    row = df_bt.iloc[i]
    dt, op, hi, lo, cl = row['date'], row['open'], row['high'], row['low'], row['close']
    c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
    
    s_flow = row['signal_etf_flow_ema3']
    s_sopr = row['signal_sopr_28ma']
    s_atr = row['signal_atr_14']
    s_vol20 = row['signal_vol_20']
    s_sma50 = row['signal_sma_50']
    
    atr_mult = 3.5 if s_vol20 > vol_75th else 3.0
    stopped_out = False
    
    # 1. Stop Loss
    if pos_weight > 0 and peak_price is not None:
        if lo <= peak_price - atr_mult * s_atr:
            stopped_out, stop_outs = True, stop_outs + 1
            exec_price = min(op, peak_price - atr_mult * s_atr)
            equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
            fee = equity * abs(pos_weight) * 0.0010
            equity -= fee; total_fees += fee; pos_weight, peak_price = 0.0, None; trade_count += 1
    elif pos_weight < 0 and trough_price is not None:
        if hi >= trough_price + atr_mult * s_atr:
            stopped_out, stop_outs = True, stop_outs + 1
            exec_price = max(op, trough_price + atr_mult * s_atr)
            equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
            fee = equity * abs(pos_weight) * 0.0005
            equity -= fee; total_fees += fee; pos_weight, trough_price = 0.0, None; trade_count += 1

    if not stopped_out:
        # 2. Raw Target with SOPR Deadband (0.01)
        if s_flow > 10.0 and s_sopr > 1.01:
            t_raw = 0.80 if (s_flow > 150.0 and s_sopr > 1.02) else 0.50
        elif s_flow < -10.0 and s_sopr < 0.99:
            t_raw = -0.50 if s_flow < -100.0 else -0.35
        elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
            t_raw = 0.10
        else:
            t_raw = prev_confirmed_target
            
        # 3. Macro Trend Filter (SMA50)
        if c_prev > s_sma50 and t_raw < 0:
            t_raw = 0.0 # No shorting in structural bull
        elif c_prev < s_sma50 and t_raw > 0:
            t_raw = 0.10 # Max 10% long in structural bear
            
        raw_target_history.append(t_raw)
        
        # 4. 2-Day Confirmation
        if len(raw_target_history) >= 2:
            recent_targets = raw_target_history[-2:]
            if all(x == recent_targets[0] for x in recent_targets):
                confirmed_target = recent_targets[0]
            else:
                confirmed_target = prev_confirmed_target
        else:
            confirmed_target = prev_confirmed_target
            
        prev_confirmed_target = confirmed_target
        
        # 5. Volatility Cap
        if s_vol20 > vol_75th:
            target_weight = min(confirmed_target, 0.40) if confirmed_target > 0 else (max(confirmed_target, -0.40) if confirmed_target < 0 else 0.0)
        else:
            target_weight = confirmed_target

        # 6. Lazy Rebalancing (15% threshold)
        if abs(target_weight - pos_weight) >= 0.15:
            turnover = abs(target_weight - pos_weight)
            fee = equity * turnover * (0.0010 if target_weight >= 0 else 0.0005)
            equity -= fee; total_fees += fee; trade_count += 1; pos_weight = target_weight
            peak_price = hi if pos_weight > 0 else None
            trough_price = lo if pos_weight < 0 else None
        else:
            if pos_weight > 0 and peak_price is not None: peak_price = max(peak_price, hi)
            elif pos_weight < 0 and trough_price is not None: trough_price = min(trough_price, lo)

        # 7. PnL
        if pos_weight > 0:
            equity += equity * pos_weight * ((cl - c_prev) / c_prev)
        elif pos_weight < 0:
            equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev) + equity * abs(pos_weight) * 0.0001
            
    equity_curve.append({
        'date': dt, 'equity': equity, 'pos_weight': pos_weight, 'btc_close': cl
    })

df_res = pd.DataFrame(equity_curve)

# Output for 2026 specifically
df_2026 = df_res[df_res['date'] >= '2026-01-01'].reset_index(drop=True)
start_eq_2026 = df_res[df_res['date'] < '2026-01-01']['equity'].iloc[-1]
end_eq_2026 = df_2026['equity'].iloc[-1]
ret_2026 = (end_eq_2026 - start_eq_2026) / start_eq_2026

df_2026['peak'] = df_2026['equity'].cummax()
mdd_2026 = ((df_2026['equity'] - df_2026['peak']) / df_2026['peak']).min()
days_2026 = len(df_2026)
ann_ret_2026 = (1 + ret_2026) ** (365.0 / days_2026) - 1
calmar_2026 = ann_ret_2026 / abs(mdd_2026) if mdd_2026 != 0 else 0

btc_start = df_2026['btc_close'].iloc[0]
btc_end = df_2026['btc_close'].iloc[-1]
btc_ret = (btc_end - btc_start) / btc_start
df_2026['btc_peak'] = df_2026['btc_close'].cummax()
btc_mdd = ((df_2026['btc_close'] - df_2026['btc_peak']) / df_2026['btc_peak']).min()

# Identify 2026 Trades
df_res['trade_flag'] = (df_res['pos_weight'] != df_res['pos_weight'].shift(1)).astype(int)
df_2026_trades = df_res[(df_res['date'] >= '2026-01-01') & (df_res['trade_flag'] == 1)]
trades_2026 = len(df_2026_trades)

print("=== 进阶降噪版 2026 年度表现 ===")
print(f"2026 初净值: ${start_eq_2026:.2f}")
print(f"2026 末净值: ${end_eq_2026:.2f}")
print(f"2026 收益率 (YTD): {ret_2026*100:.2f}%")
print(f"2026 最大回撤 (MDD): {mdd_2026*100:.2f}%")
print(f"2026 卡玛比率 (Calmar): {calmar_2026:.2f}")
print(f"2026 交易次数: {trades_2026} 次")
print("--------------------------------")
print(f"同期 BTC 收益率: {btc_ret*100:.2f}%")
print(f"同期 BTC 最大回撤: {btc_mdd*100:.2f}%")

# Quarters
df_2026['q'] = df_2026['date'].dt.to_period('Q')
for q in df_2026['q'].unique():
    dq = df_2026[df_2026['q'] == q]
    q_st = df_res[df_res['date'] < dq['date'].min()]['equity'].iloc[-1]
    q_ed = dq['equity'].iloc[-1]
    q_ret = (q_ed - q_st) / q_st
    q_mdd = ((dq['equity'] - dq['equity'].cummax()) / dq['equity'].cummax()).min()
    
    b_st = df_res[df_res['date'] < dq['date'].min()]['btc_close'].iloc[-1]
    b_ed = dq['btc_close'].iloc[-1]
    b_ret = (b_ed - b_st) / b_st
    print(f"[{q}] 策略: {q_ret*100:+.2f}% (回撤 {q_mdd*100:.2f}%) | BTC: {b_ret*100:+.2f}%")
