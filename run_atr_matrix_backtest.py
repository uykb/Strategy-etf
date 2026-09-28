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
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
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
df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)

# Shift 1D
df['signal_flow'] = df['etf_flow'].shift(1)
df['signal_sopr'] = df['sopr_28ma'].shift(1)
df['signal_atr'] = df['atr_14'].shift(1)
df['signal_vol20'] = df['vol_20'].shift(1)

bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol20'].quantile(0.75)

def run_bracket_sim(sl_mult=1.5, tp_mult=3.0):
    initial_capital = 1000.0
    equity = initial_capital
    pos_state = 0 # 0: FLAT, +1: LONG, -1: SHORT
    pos_weight = 0.0

    entry_price, sl_price, tp_price = None, None, None
    win_count, loss_count, total_fees = 0, 0, 0.0
    equity_curve = []

    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        dt, op, hi, lo, cl = row['date'], row['open'], row['high'], row['low'], row['close']
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
        s_flow, s_sopr, s_atr, s_vol20 = row['signal_flow'], row['signal_sopr'], row['signal_atr'], row['signal_vol20']

        # Check Active Position SL/TP
        if pos_state != 0:
            exited = False
            exec_price = None

            if pos_state > 0: # Long
                if lo <= sl_price:
                    exited = True
                    exec_price = min(op, sl_price)
                    loss_count += 1
                elif hi >= tp_price:
                    exited = True
                    exec_price = max(op, tp_price)
                    win_count += 1

                if exited:
                    equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0010
                    equity -= fee; total_fees += fee
                    pos_state, pos_weight, entry_price = 0, 0.0, None

            elif pos_state < 0: # Short
                if hi >= sl_price:
                    exited = True
                    exec_price = max(op, sl_price)
                    loss_count += 1
                elif lo <= tp_price:
                    exited = True
                    exec_price = min(op, tp_price)
                    win_count += 1

                if exited:
                    equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0005
                    equity -= fee; total_fees += fee
                    pos_state, pos_weight, entry_price = 0, 0.0, None

        # Check Signal when FLAT
        if pos_state == 0:
            if s_flow > 0 and s_sopr > 1.0:
                pos_state = 1
                pos_weight = 0.40 if s_vol20 > vol_75th else 0.70
                entry_price = op
                sl_price = entry_price - sl_mult * s_atr
                tp_price = entry_price + tp_mult * s_atr
                fee = equity * pos_weight * 0.0010
                equity -= fee; total_fees += fee

            elif s_flow < 0 and s_sopr < 1.0:
                pos_state = -1
                pos_weight = -0.40 if s_vol20 > vol_75th else -0.45
                entry_price = op
                sl_price = entry_price + sl_mult * s_atr
                tp_price = entry_price - tp_mult * s_atr
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee; total_fees += fee

        # Daily PnL
        if pos_state > 0:
            equity += equity * pos_weight * ((cl - c_prev) / c_prev)
        elif pos_state < 0:
            equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev) + equity * abs(pos_weight) * 0.0001

        equity_curve.append({'date': dt, 'equity': equity})

    df_res = pd.DataFrame(equity_curve)
    ret = (equity - initial_capital) / initial_capital
    df_res['peak'] = df_res['equity'].cummax()
    mdd = ((df_res['equity'] - df_res['peak']) / df_res['peak']).min()
    win_rate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0
    total_trades = win_count + loss_count

    # 2026 YTD
    df_2026 = df_res[df_res['date'] >= '2026-01-01'].reset_index(drop=True)
    st_2026 = df_res[df_res['date'] < '2026-01-01']['equity'].iloc[-1]
    ed_2026 = df_2026['equity'].iloc[-1]
    ret_2026 = (ed_2026 - st_2026) / st_2026

    return {
        'sl_mult': sl_mult,
        'tp_mult': tp_mult,
        'rr_ratio': tp_mult / sl_mult,
        'final_equity': equity,
        'ret': ret,
        'ret_2026': ret_2026,
        'mdd': mdd,
        'win_rate': win_rate,
        'trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'fees': total_fees
    }

sl_list = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
tp_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

matrix_results = []
for sl in sl_list:
    for tp in tp_list:
        res = run_bracket_sim(sl, tp)
        matrix_results.append(res)

df_matrix = pd.DataFrame(matrix_results)
df_matrix = df_matrix.sort_values('ret', ascending=False).reset_index(drop=True)

print("=== ATR TP/SL MULTIPLIER MATRIX BACKTEST RESULTS ===")
print(f"Total Combinations Tested: {len(df_matrix)}")
print("\nTop 15 Best Performing Combinations:")
cols = ['sl_mult', 'tp_mult', 'rr_ratio', 'final_equity', 'ret', 'ret_2026', 'mdd', 'win_rate', 'trades']
df_print = df_matrix[cols].copy()
df_print['ret'] = (df_print['ret'] * 100).map('{:+.2f}%'.format)
df_print['ret_2026'] = (df_print['ret_2026'] * 100).map('{:+.2f}%'.format)
df_print['mdd'] = (df_print['mdd'] * 100).map('{:.2f}%'.format)
df_print['win_rate'] = df_print['win_rate'].map('{:.2f}%'.format)
df_print['final_equity'] = df_print['final_equity'].map('${:,.2f}'.format)
df_print['rr_ratio'] = df_print['rr_ratio'].map('{:.2f}:1'.format)
print(df_print.head(15).to_string(index=False))

# Export full matrix to CSV
df_matrix.to_csv('atr_tpsl_matrix_results.csv', index=False)
