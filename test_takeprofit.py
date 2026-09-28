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
prev_close = df['close'].shift(1)
df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
df['atr_14'] = df['tr'].rolling(14).mean()
cost_basis = df['close'].ewm(span=28, adjust=False).mean()
df['sopr_28ma'] = (df['close'] / cost_basis).rolling(28).mean()

df['ret'] = df['close'].pct_change()
df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(365)

# Calculate consecutive positive/negative flow days
consec_inflow = []
consec_outflow = []
in_cnt, out_cnt = 0, 0
for flow in df['etf_flow']:
    if flow > 5.0:
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

# Shift 1D
df['signal_consec_inflow'] = df['consec_inflow'].shift(1)
df['signal_consec_outflow'] = df['consec_outflow'].shift(1)
df['signal_sopr'] = df['sopr_28ma'].shift(1)
df['signal_atr'] = df['atr_14'].shift(1)
df['signal_vol20'] = df['vol_20'].shift(1)

bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol20'].quantile(0.75)

def run_takeprofit_sim(tp_mode='none', tp_pct=0.08, trailing_tp_activation=0.05, trailing_tp_delta=0.03):
    initial_capital = 1000.0
    equity = initial_capital
    pos_weight = 0.0
    prev_target = 0.0
    peak_price, trough_price = None, None
    entry_price = None
    max_unrealized_profit = 0.0
    
    trade_count, stop_outs, take_profits, total_fees = 0, 0, 0, 0.0
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
        exited = False
        
        # 1. Check Trailing Stop Loss
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - atr_mult * s_atr
            if lo <= stop_price:
                exited, stop_outs = True, stop_outs + 1
                exec_price = min(op, stop_price)
                equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee; total_fees += fee; pos_weight, peak_price, entry_price = 0.0, None, None; trade_count += 1

        elif pos_weight < 0 and trough_price is not None:
            stop_price = trough_price + atr_mult * s_atr
            if hi >= stop_price:
                exited, stop_outs = True, stop_outs + 1
                exec_price = max(op, stop_price)
                equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee; total_fees += fee; pos_weight, trough_price, entry_price = 0.0, None, None; trade_count += 1

        # 2. Check Take-Profit Logic (If not stopped out)
        if not exited and pos_weight != 0 and entry_price is not None:
            if pos_weight > 0:
                current_profit = (hi - entry_price) / entry_price
                max_unrealized_profit = max(max_unrealized_profit, current_profit)
                
                # Mode A: Fixed Percentage TP
                if tp_mode == 'fixed' and current_profit >= tp_pct:
                    exited, take_profits = True, take_profits + 1
                    exec_price = entry_price * (1.0 + tp_pct)
                    equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0010
                    equity -= fee; total_fees += fee; pos_weight, peak_price, entry_price = 0.0, None, None; trade_count += 1
                
                # Mode B: Trailing Profit Lock TP (Activation + Pullback)
                elif tp_mode == 'trailing' and max_unrealized_profit >= trailing_tp_activation:
                    pullback_trigger_price = entry_price * (1.0 + max_unrealized_profit - trailing_tp_delta)
                    if lo <= pullback_trigger_price:
                        exited, take_profits = True, take_profits + 1
                        exec_price = min(op, pullback_trigger_price)
                        equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                        fee = equity * abs(pos_weight) * 0.0010
                        equity -= fee; total_fees += fee; pos_weight, peak_price, entry_price = 0.0, None, None; trade_count += 1

            elif pos_weight < 0:
                current_profit = (entry_price - lo) / entry_price
                max_unrealized_profit = max(max_unrealized_profit, current_profit)
                
                # Mode A: Fixed Percentage TP
                if tp_mode == 'fixed' and current_profit >= tp_pct:
                    exited, take_profits = True, take_profits + 1
                    exec_price = entry_price * (1.0 - tp_pct)
                    equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0005
                    equity -= fee; total_fees += fee; pos_weight, trough_price, entry_price = 0.0, None, None; trade_count += 1
                
                # Mode B: Trailing Profit Lock TP
                elif tp_mode == 'trailing' and max_unrealized_profit >= trailing_tp_activation:
                    pullback_trigger_price = entry_price * (1.0 - max_unrealized_profit + trailing_tp_delta)
                    if hi >= pullback_trigger_price:
                        exited, take_profits = True, take_profits + 1
                        exec_price = max(op, pullback_trigger_price)
                        equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                        fee = equity * abs(pos_weight) * 0.0005
                        equity -= fee; total_fees += fee; pos_weight, trough_price, entry_price = 0.0, None, None; trade_count += 1

        # 3. Entry / Signal Logic
        if not exited:
            N_inflow, N_outflow = 3, 3
            if s_in >= N_inflow:
                target_raw = 0.70
            elif s_out >= N_outflow:
                target_raw = -0.45
            elif s_in == 0 and s_out == 0:
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
                fee = equity * turnover * (0.0010 if target_weight >= 0 else 0.0005)
                equity -= fee; total_fees += fee; trade_count += 1
                pos_weight = target_weight
                entry_price = cl
                max_unrealized_profit = 0.0
                peak_price = hi if pos_weight > 0 else None
                trough_price = lo if pos_weight < 0 else None
            else:
                if pos_weight > 0 and peak_price is not None: peak_price = max(peak_price, hi)
                elif pos_weight < 0 and trough_price is not None: trough_price = min(trough_price, lo)

            if pos_weight > 0:
                equity += equity * pos_weight * ((cl - c_prev) / c_prev)
            elif pos_weight < 0:
                equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev)
                equity += equity * abs(pos_weight) * 0.0001

        equity_curve.append(equity)

    df_eval = pd.DataFrame({'equity': equity_curve})
    ret = (equity - initial_capital) / initial_capital
    ann_ret = (1 + ret) ** (365.0 / len(df_bt)) - 1
    df_eval['peak'] = df_eval['equity'].cummax()
    mdd = ((df_eval['equity'] - df_eval['peak']) / df_eval['peak']).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    daily_ret = df_eval['equity'].pct_change().fillna(0.0)
    sharpe = (daily_ret.mean() - 0.02/365) / daily_ret.std() * np.sqrt(365)

    return {
        'mode': tp_mode,
        'final_equity': equity,
        'ret': ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'calmar': calmar,
        'sharpe': sharpe,
        'trades': trade_count,
        'stop_outs': stop_outs,
        'take_profits': take_profits,
        'fees': total_fees
    }

print("=== TAKE-PROFIT EXPERIMENT ON CONSECUTIVE ETF FLOW STRATEGY (N=3) ===")
r0 = run_takeprofit_sim(tp_mode='none')
print(f"1. No Take Profit: Ret={r0['ret']*100:+.2f}%, MDD={r0['mdd']*100:.2f}%, Sharpe={r0['sharpe']:.2f}, Trades={r0['trades']}, TP_Count={r0['take_profits']}")

r1 = run_takeprofit_sim(tp_mode='fixed', tp_pct=0.06)
print(f"2. Fixed TP (+6%): Ret={r1['ret']*100:+.2f}%, MDD={r1['mdd']*100:.2f}%, Sharpe={r1['sharpe']:.2f}, Trades={r1['trades']}, TP_Count={r1['take_profits']}")

r2 = run_takeprofit_sim(tp_mode='fixed', tp_pct=0.10)
print(f"3. Fixed TP (+10%): Ret={r2['ret']*100:+.2f}%, MDD={r2['mdd']*100:.2f}%, Sharpe={r2['sharpe']:.2f}, Trades={r2['trades']}, TP_Count={r2['take_profits']}")

r3 = run_takeprofit_sim(tp_mode='trailing', trailing_tp_activation=0.05, trailing_tp_delta=0.02)
print(f"4. Trailing TP (Activate 5%, Pullback 2%): Ret={r3['ret']*100:+.2f}%, MDD={r3['mdd']*100:.2f}%, Sharpe={r3['sharpe']:.2f}, Trades={r3['trades']}, TP_Count={r3['take_profits']}")

r4 = run_takeprofit_sim(tp_mode='trailing', trailing_tp_activation=0.08, trailing_tp_delta=0.03)
print(f"5. Trailing TP (Activate 8%, Pullback 3%): Ret={r4['ret']*100:+.2f}%, MDD={r4['mdd']*100:.2f}%, Sharpe={r4['sharpe']:.2f}, Trades={r4['trades']}, TP_Count={r4['take_profits']}")
