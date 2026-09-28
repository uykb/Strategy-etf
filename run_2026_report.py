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
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)

def fetch_farside_etf_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get('https://farside.co.uk/bitcoin-etf-flow-all-data/', headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    parsed = []
    table = soup.find_all('table')[0]
    for r in table.find_all('tr'):
        cols = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
        if len(cols) >= 2 and re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', cols[0]):
            val_str = cols[-1].replace('$', '').replace(',', '').strip()
            val = -float(val_str[1:-1]) if val_str.startswith('(') and val_str.endswith(')') else (float(val_str) if val_str not in ['-', '', 'NaN'] else 0.0)
            parsed.append({'date_str': cols[0], 'etf_flow': val})
    df_etf = pd.DataFrame(parsed)
    df_etf['date'] = pd.to_datetime(df_etf['date_str'], format='%d %b %Y', errors='coerce')
    return df_etf.dropna(subset=['date']).sort_values('date')[['date', 'etf_flow']].drop_duplicates(subset=['date']).reset_index(drop=True)

df_btc = fetch_btc_data()
df_etf = fetch_farside_etf_data()

df = pd.merge(df_btc, df_etf, on='date', how='left')
df['etf_flow'] = df['etf_flow'].fillna(0.0)

# Indicators
df['etf_flow_ema3'] = df['etf_flow'].ewm(span=3, adjust=False).mean()
prev_close = df['close'].shift(1)
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
df['atr_14'] = df['tr'].rolling(14).mean()
cost_basis = df['close'].ewm(span=28, adjust=False).mean()
df['sopr_28ma'] = (df['close'] / cost_basis).rolling(28).mean()

df['ret'] = df['close'].pct_change()
df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(365)

# Shift 1D
df['signal_flow'] = df['etf_flow_ema3'].shift(1)
df['signal_sopr'] = df['sopr_28ma'].shift(1)
df['signal_atr'] = df['atr_14'].shift(1)
df['signal_vol20'] = df['vol_20'].shift(1)

bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol20'].quantile(0.75)

initial_capital = 1000.0
equity = initial_capital
pos_weight = 0.0
prev_target = 0.0
peak_price, trough_price = None, None
trade_count, stop_outs, total_fees, funding_earned_total = 0, 0, 0.0, 0.0

equity_curve = []

for i in range(len(df_bt)):
    row = df_bt.iloc[i]
    dt, op, hi, lo, cl = row['date'], row['open'], row['high'], row['low'], row['close']
    c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
    s_flow, s_sopr, s_atr, s_vol20 = row['signal_flow'], row['signal_sopr'], row['signal_atr'], row['signal_vol20']

    atr_mult = 3.5 if s_vol20 > vol_75th else 3.0
    stopped_out = False

    if pos_weight > 0 and peak_price is not None:
        if lo <= peak_price - atr_mult * s_atr:
            stopped_out = True
            stop_outs += 1
            exec_price = min(op, peak_price - atr_mult * s_atr)
            equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
            fee = equity * abs(pos_weight) * 0.0010
            equity -= fee
            total_fees += fee
            pos_weight, peak_price = 0.0, None
            trade_count += 1
    elif pos_weight < 0 and trough_price is not None:
        if hi >= trough_price + atr_mult * s_atr:
            stopped_out = True
            stop_outs += 1
            exec_price = max(op, trough_price + atr_mult * s_atr)
            equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
            fee = equity * abs(pos_weight) * 0.0005
            equity -= fee
            total_fees += fee
            pos_weight, trough_price = 0.0, None
            trade_count += 1

    if not stopped_out:
        if s_flow > 10.0 and s_sopr > 1.0:
            target_raw = 0.80 if s_flow > 150.0 and s_sopr > 1.02 else 0.50
        elif s_flow < -10.0 and s_sopr < 1.0:
            target_raw = -0.50 if s_flow < -100.0 else -0.35
        elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
            target_raw = 0.10
        else:
            target_raw = prev_target

        prev_target = target_raw

        if s_vol20 > vol_75th:
            target_weight = min(target_raw, 0.40) if target_raw > 0 else (max(target_raw, -0.40) if target_raw < 0 else 0.0)
        else:
            target_weight = target_raw

        if abs(target_weight - pos_weight) >= 0.10:
            turnover = abs(target_weight - pos_weight)
            fee_rate = 0.0010 if target_weight >= 0 else 0.0005
            fee = equity * turnover * fee_rate
            equity -= fee
            total_fees += fee
            trade_count += 1
            pos_weight = target_weight
            peak_price = hi if pos_weight > 0 else None
            trough_price = lo if pos_weight < 0 else None
        else:
            if pos_weight > 0 and peak_price is not None: peak_price = max(peak_price, hi)
            elif pos_weight < 0 and trough_price is not None: trough_price = min(trough_price, lo)

        if pos_weight > 0:
            equity += equity * pos_weight * ((cl - c_prev) / c_prev)
        elif pos_weight < 0:
            equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev)
            f_earn = equity * abs(pos_weight) * 0.0001
            equity += f_earn
            funding_earned_total += f_earn

    btc_start = df_bt.iloc[0]['close']
    btc_bh_equity = initial_capital * (cl / btc_start)

    equity_curve.append({
        'date': dt,
        'equity': equity,
        'pos_weight': pos_weight,
        'btc_close': cl,
        'btc_bh_equity': btc_bh_equity,
        'etf_flow': row['etf_flow'],
        'etf_flow_ema3': row['etf_flow_ema3'],
        'sopr_28ma': row['sopr_28ma'],
        'atr_14': row['atr_14'],
        'vol_20': row['vol_20']
    })

df_res = pd.DataFrame(equity_curve)

# Filter specifically for THIS YEAR (2026-01-01 to 2026-09-27)
df_2026 = df_res[df_res['date'] >= '2026-01-01'].copy().reset_index(drop=True)

# 2026 YTD metrics
start_eq_2026 = df_res[df_res['date'] < '2026-01-01']['equity'].iloc[-1]
end_eq_2026 = df_2026['equity'].iloc[-1]
ret_2026 = (end_eq_2026 - start_eq_2026) / start_eq_2026

days_2026 = (df_2026['date'].max() - df_2026['date'].min()).days + 1
ann_ret_2026 = (1 + ret_2026) ** (365.0 / days_2026) - 1

df_2026['peak'] = df_2026['equity'].cummax()
df_2026['mdd'] = (df_2026['equity'] - df_2026['peak']) / df_2026['peak']
mdd_2026 = df_2026['mdd'].min()
calmar_2026 = ann_ret_2026 / abs(mdd_2026) if mdd_2026 != 0 else 0

daily_ret_2026 = df_2026['equity'].pct_change().fillna(0.0)
sharpe_2026 = (daily_ret_2026.mean() - 0.02/365) / daily_ret_2026.std() * np.sqrt(365)

# Benchmark 2026 BTC Buy & Hold
btc_start_2026 = df_2026['btc_close'].iloc[0]
btc_end_2026 = df_2026['btc_close'].iloc[-1]
btc_ret_2026 = (btc_end_2026 - btc_start_2026) / btc_start_2026
df_2026['btc_peak'] = df_2026['btc_close'].cummax()
btc_mdd_2026 = ((df_2026['btc_close'] - df_2026['btc_peak']) / df_2026['btc_peak']).min()

print("=== 2026 今年年度 (YTD) 完整表现报告 ===")
print(f"统计区间: 2026-01-01 至 2026-09-27 ({days_2026} 天)")
print(f"2026 年初净值: ${start_eq_2026:.2f}")
print(f"2026 年末净值: ${end_eq_2026:.2f}")
print(f"2026 累计收益率 (YTD Return): {ret_2026*100:.2f}%")
print(f"2026 年化收益率 (CAGR): {ann_ret_2026*100:.2f}%")
print(f"2026 最大回撤 (MDD): {mdd_2026*100:.2f}%")
print(f"2026 卡玛比率 (Calmar): {calmar_2026:.2f}")
print(f"2026 夏普比率 (Sharpe): {sharpe_2026:.2f}")
print("---------------------------------------")
print("2026 比特币基准对比 (BTC Buy & Hold):")
print(f"2026 BTC 年初价格: ${btc_start_2026:,.2f}")
print(f"2026 BTC 最新价格: ${btc_end_2026:,.2f}")
print(f"2026 BTC 累计收益率: {btc_ret_2026*100:.2f}%")
print(f"2026 BTC 最大回撤: {btc_mdd_2026*100:.2f}%")

# 2026 Quarterly Breakdown
df_2026['quarter'] = df_2026['date'].dt.to_period('Q')
quarters_2026 = df_2026['quarter'].unique()

print("\n=== 2026 年分季度拆解 ===")
for q in quarters_2026:
    df_q = df_2026[df_2026['quarter'] == q]
    q_start = df_res[df_res['date'] < df_q['date'].min()]['equity'].iloc[-1]
    q_end = df_q['equity'].iloc[-1]
    q_ret = (q_end - q_start) / q_start
    q_peak = df_q['equity'].cummax()
    q_mdd = ((df_q['equity'] - q_peak) / q_peak).min()
    
    # BTC quarter ret
    btc_q_start = df_btc[df_btc['date'] < df_q['date'].min()]['close'].iloc[-1] if len(df_btc[df_btc['date'] < df_q['date'].min()])>0 else df_q['btc_close'].iloc[0]
    btc_q_end = df_q['btc_close'].iloc[-1]
    btc_q_ret = (btc_q_end - btc_q_start) / btc_q_start
    
    print(f"季度: {q} | 策略收益: {q_ret*100:+.2f}% | 策略最大回撤: {q_mdd*100:.2f}% | 期初: ${q_start:.2f} -> 期末: ${q_end:.2f} (对比 BTC 收益: {btc_q_ret*100:+.2f}%)")
