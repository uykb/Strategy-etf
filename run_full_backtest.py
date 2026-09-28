import urllib.request
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import re
import os
import matplotlib.pyplot as plt

# Set font & style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
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

def run_strategy_backtest(long_target=0.70, short_target=-0.45, contra_target=0.15, vol_cap=0.40):
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()

    # Merge BTC price and ETF flow data on date
    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)

    # 1. ATR(14)
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr_14'] = df['tr'].rolling(14).mean()

    # 2. SOPR and SOPR 28MA
    cost_basis = df['close'].ewm(span=28, adjust=False).mean()
    df['sopr'] = df['close'] / cost_basis
    df['sopr_28ma'] = df['sopr'].rolling(28).mean()

    # 3. 20-day annualized volatility
    df['ret'] = df['close'].pct_change()
    df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(365)

    # 4. Shift 1D to prevent lookahead bias (前视偏差)
    df['signal_flow'] = df['etf_flow'].shift(1)
    df['signal_sopr'] = df['sopr_28ma'].shift(1)
    df['signal_atr'] = df['atr_14'].shift(1)
    df['signal_vol20'] = df['vol_20'].shift(1)

    # Filter backtest period: 2025-01-01 to 2026-09-27
    bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
    df_bt = df[bt_mask].copy().reset_index(drop=True)
    vol_75th = df_bt['signal_vol20'].quantile(0.75)

    # Simulation setup
    initial_capital = 1000.0
    equity = initial_capital
    pos_weight = 0.0
    prev_target_raw = 0.0
    peak_price = None
    trough_price = None
    
    trade_count = 0
    total_funding_earned = 0.0
    total_trade_fees = 0.0
    stop_outs_count = 0
    
    equity_curve = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        dt = row['date']
        op = row['open']
        hi = row['high']
        lo = row['low']
        cl = row['close']
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
        
        s_flow = row['signal_flow']
        s_sopr = row['signal_sopr']
        s_atr = row['signal_atr']
        s_vol = row['signal_vol20']
        
        stopped_out = False
        
        # Check Trailing Stop Loss (3x ATR) on active position
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - 3.0 * s_atr
            if lo <= stop_price:
                stopped_out = True
                stop_outs_count += 1
                exec_price = min(op, stop_price)
                day_ret = (exec_price - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee
                total_trade_fees += fee
                pos_weight = 0.0
                peak_price = None
                trade_count += 1
                
        elif pos_weight < 0 and trough_price is not None:
            stop_price = trough_price + 3.0 * s_atr
            if hi >= stop_price:
                stopped_out = True
                stop_outs_count += 1
                exec_price = max(op, stop_price)
                day_ret = (c_prev - exec_price) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee
                total_trade_fees += fee
                pos_weight = 0.0
                trough_price = None
                trade_count += 1

        # Evaluate signals & rebalance if not stopped out
        if not stopped_out:
            if s_flow > 0 and s_sopr > 1.0:
                target_raw = long_target
            elif s_flow < 0 and s_sopr < 1.0:
                target_raw = short_target
            elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
                target_raw = contra_target
            elif s_flow == 0:
                target_raw = prev_target_raw
            else:
                target_raw = prev_target_raw
                
            prev_target_raw = target_raw
            
            # Volatility filter cap (20d vol > 75th percentile -> max pos <= 40%)
            if s_vol > vol_75th:
                if target_raw > 0:
                    target_weight = min(target_raw, vol_cap)
                elif target_raw < 0:
                    target_weight = max(target_raw, -vol_cap)
                else:
                    target_weight = 0.0
            else:
                target_weight = target_raw
                
            # Rebalance trade check
            if abs(target_weight - pos_weight) > 1e-4:
                turnover = abs(target_weight - pos_weight)
                fee_rate = 0.0010 if target_weight >= 0 else 0.0005
                fee = equity * turnover * fee_rate
                equity -= fee
                total_trade_fees += fee
                trade_count += 1
                pos_weight = target_weight
                
                if pos_weight > 0:
                    peak_price = hi
                    trough_price = None
                elif pos_weight < 0:
                    trough_price = lo
                    peak_price = None
                else:
                    peak_price = None
                    trough_price = None
            else:
                if pos_weight > 0 and peak_price is not None:
                    peak_price = max(peak_price, hi)
                elif pos_weight < 0 and trough_price is not None:
                    trough_price = min(trough_price, lo)

            # Daily PnL
            if pos_weight > 0:
                day_ret = (cl - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
            elif pos_weight < 0:
                day_ret = (c_prev - cl) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                funding_earned = equity * abs(pos_weight) * 0.0001
                equity += funding_earned
                total_funding_earned += funding_earned

        btc_start = df_bt.iloc[0]['close']
        btc_bh_equity = initial_capital * (cl / btc_start)

        equity_curve.append({
            'date': dt,
            'equity': equity,
            'pos_weight': pos_weight,
            'btc_close': cl,
            'btc_bh_equity': btc_bh_equity,
            'etf_flow': row['etf_flow'],
            'sopr_28ma': row['sopr_28ma'],
            'atr_14': row['atr_14'],
            'vol_20': row['vol_20']
        })

    df_res = pd.DataFrame(equity_curve)
    
    total_days = (df_bt['date'].max() - df_bt['date'].min()).days + 1
    total_return = (equity - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
    
    df_res['peak'] = df_res['equity'].cummax()
    df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak']
    max_drawdown = df_res['drawdown'].min()
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    df_res['daily_ret'] = df_res['equity'].pct_change().fillna(0.0)
    rf_daily = 0.02 / 365.0
    sharpe_ratio = (df_res['daily_ret'].mean() - rf_daily) / df_res['daily_ret'].std() * np.sqrt(365) if df_res['daily_ret'].std() > 0 else 0

    stats = {
        'initial_capital': initial_capital,
        'final_equity': equity,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': trade_count,
        'stop_outs_count': stop_outs_count,
        'total_funding_earned': total_funding_earned,
        'total_trade_fees': total_trade_fees,
        'total_days': total_days,
        'vol_75th': vol_75th
    }
    
    return df_res, stats

# Run primary scenario (Long 70%, Short 45%, Contra 15%)
df_res, stats = run_strategy_backtest(long_target=0.70, short_target=-0.45, contra_target=0.15)

# Plotting performance chart
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1]})

# Equity Curve
ax1.plot(df_res['date'], df_res['equity'], label='BTC ETF+SOPR Strategy', color='#1f77b4', linewidth=2)
ax1.plot(df_res['date'], df_res['btc_bh_equity'], label='BTC Buy & Hold', color='#7f7f7f', linestyle='--', linewidth=1.5)
ax1.set_title('BTC ETF Flow & SOPR Strategy Backtest (2025-01-01 to 2026-09-27)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Net Value ($)', fontsize=12)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)

# Drawdown Plot
ax2.plot(df_res['date'], df_res['drawdown'] * 100, label='Strategy Drawdown (%)', color='#d62728', linewidth=1.5)
ax2.fill_between(df_res['date'], df_res['drawdown'] * 100, 0, color='#d62728', alpha=0.2)
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)

# Position Weight Plot
ax3.plot(df_res['date'], df_res['pos_weight'] * 100, label='Position Exposure (%)', color='#2ca02c', linewidth=1.2)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
ax3.set_ylabel('Position (%)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save plot to scratch and workspace
plot_path_workspace = r'c:\Users\uykb\Documents\git\Strategy-etf\backtest_result.png'
plt.savefig(plot_path_workspace, dpi=150)
print(f"Chart saved to {plot_path_workspace}")
