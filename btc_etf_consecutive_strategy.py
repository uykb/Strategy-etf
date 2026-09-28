"""
BTC ETF 连续流入/流出开仓策略 (Consecutive ETF Flow Strategy)
规则:
1. 连续 N 天 ETF 净流入 -> 开/加多仓 (现货 70%)
2. 连续 N 天 ETF 净流出 -> 开/加空仓 (永续 2倍杠杆名义 -45%)
3. 流入/流出中断 -> 平仓至现金/轻仓 (10%)
4. 风控: 3x ATR 移动止损 + 20日波动率 > 75分位压缩仓位至 40%
"""

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

def run_consecutive_strategy(N_inflow=3, N_outflow=3, long_weight=0.70, short_weight=-0.45):
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()

    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)

    # 1. 计算连续流入/流出天数
    consec_inflow, consec_outflow = [], []
    in_cnt, out_cnt = 0, 0
    for flow in df['etf_flow']:
        if flow > 5.0: # $5M 过滤门槛
            in_cnt += 1
            out_cnt = 0
        elif flow < -5.0:
            out_cnt += 1
            in_cnt = 0
        else:
            in_cnt = 0
            out_cnt = 0
        consec_inflow.append(in_cnt)
        consec_outflow.append(out_cnt)

    df['consec_inflow'] = consec_inflow
    df['consec_outflow'] = consec_outflow

    # 2. 指标与 ATR
    prev_close = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)

    # 3. 严格 shift(1D) 消除前视偏差
    df['signal_consec_inflow'] = df['consec_inflow'].shift(1)
    df['signal_consec_outflow'] = df['consec_outflow'].shift(1)
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
    trade_count, stop_outs, total_fees, funding_earned = 0, 0, 0.0, 0.0
    equity_curve = []

    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        dt, op, hi, lo, cl = row['date'], row['open'], row['high'], row['low'], row['close']
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
        
        s_in = row['signal_consec_inflow']
        s_out = row['signal_consec_outflow']
        s_atr = row['signal_atr']
        s_vol20 = row['signal_vol20']
        
        atr_mult = 3.5 if s_vol20 > vol_75th else 3.0
        stopped_out = False
        
        # 止损逻辑
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - atr_mult * s_atr
            if lo <= stop_price:
                stopped_out = True; stop_outs += 1
                exec_price = min(op, stop_price)
                equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee; total_fees += fee
                pos_weight, peak_price = 0.0, None; trade_count += 1
        elif pos_weight < 0 and trough_price is not None:
            stop_price = trough_price + atr_mult * s_atr
            if hi >= stop_price:
                stopped_out = True; stop_outs += 1
                exec_price = max(op, stop_price)
                equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee; total_fees += fee
                pos_weight, trough_price = 0.0, None; trade_count += 1

        # 连续流入/流出信号
        if not stopped_out:
            if s_in >= N_inflow:
                target_raw = long_weight
            elif s_out >= N_outflow:
                target_raw = short_weight
            elif s_in == 0 and s_out == 0:
                target_raw = 0.10 # 中断离场
            else:
                target_raw = prev_target
                
            prev_target = target_raw
            
            # 波动率限制
            if s_vol20 > vol_75th:
                target_weight = min(target_raw, 0.40) if target_raw > 0 else (max(target_raw, -0.40) if target_raw < 0 else 0.0)
            else:
                target_weight = target_raw

            # 调仓
            if abs(target_weight - pos_weight) >= 0.10:
                turnover = abs(target_weight - pos_weight)
                fee = equity * turnover * (0.0010 if target_weight >= 0 else 0.0005)
                equity -= fee; total_fees += fee; trade_count += 1
                pos_weight = target_weight
                peak_price = hi if pos_weight > 0 else None
                trough_price = lo if pos_weight < 0 else None
            else:
                if pos_weight > 0 and peak_price is not None: peak_price = max(peak_price, hi)
                elif pos_weight < 0 and trough_price is not None: trough_price = min(trough_price, lo)

            # PnL
            if pos_weight > 0:
                equity += equity * pos_weight * ((cl - c_prev) / c_prev)
            elif pos_weight < 0:
                equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev)
                funding_earned += equity * abs(pos_weight) * 0.0001
                equity += equity * abs(pos_weight) * 0.0001

        equity_curve.append({'date': dt, 'equity': equity, 'pos_weight': pos_weight})

    df_res = pd.DataFrame(equity_curve)
    ret = (equity - initial_capital) / initial_capital
    mdd = ((df_res['equity'] - df_res['equity'].cummax()) / df_res['equity'].cummax()).min()
    
    print(f"=== BTC ETF 连续 {N_inflow} 天流入/流出策略回测 ===")
    print(f"最终净值: ${equity:.2f} | 累计收益率: {ret*100:+.2f}% | 最大回撤: {mdd*100:.2f}% | 总交易数: {trade_count} 次")

if __name__ == '__main__':
    run_consecutive_strategy(N_inflow=3, N_outflow=3)
