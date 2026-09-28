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

def run_backtest():
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

    # 4. Shift 1D to prevent lookahead bias
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
    pos_weight = 0.0 # positive for spot long, negative for perp short
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
        
        # Check Trailing Stop Loss on existing position
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - 3.0 * s_atr
            if lo <= stop_price:
                # Long trailing stop hit
                stopped_out = True
                stop_outs_count += 1
                exec_price = min(op, stop_price)
                day_ret = (exec_price - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
                # Closing fee (spot 0.1%)
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee
                total_trade_fees += fee
                pos_weight = 0.0
                peak_price = None
                trade_count += 1
                
        elif pos_weight < 0 and trough_price is not None:
            stop_price = trough_price + 3.0 * s_atr
            if hi >= stop_price:
                # Short trailing stop hit
                stopped_out = True
                stop_outs_count += 1
                exec_price = max(op, stop_price)
                day_ret = (c_prev - exec_price) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                # Closing fee (perp 0.05%)
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee
                total_trade_fees += fee
                pos_weight = 0.0
                trough_price = None
                trade_count += 1

        # If not stopped out, evaluate signals & rebalance
        if not stopped_out:
            # Signal classification
            if s_flow > 0 and s_sopr > 1.0:
                target_raw = 0.70 # Spot Long 70%
            elif s_flow < 0 and s_sopr < 1.0:
                target_raw = -0.45 # Short Perp 45% nominal
            elif (s_flow > 0 and s_sopr < 1.0) or (s_flow < 0 and s_sopr > 1.0):
                target_raw = 0.15 # Contradiction signal: max 20% (use 15% spot)
            elif s_flow == 0:
                target_raw = prev_target_raw
            else:
                target_raw = prev_target_raw
                
            prev_target_raw = target_raw
            
            # Volatility filter cap (20d vol > 75th percentile -> max pos <= 40%)
            if s_vol > vol_75th:
                if target_raw > 0:
                    target_weight = min(target_raw, 0.40)
                elif target_raw < 0:
                    target_weight = max(target_raw, -0.40)
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
                
                # Reset stop loss reference prices on new entry/change
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
                # Update peak/trough for active position
                if pos_weight > 0 and peak_price is not None:
                    peak_price = max(peak_price, hi)
                elif pos_weight < 0 and trough_price is not None:
                    trough_price = min(trough_price, lo)

            # Daily PnL for active position
            if pos_weight > 0:
                day_ret = (cl - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
            elif pos_weight < 0:
                day_ret = (c_prev - cl) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                # Daily funding fee earned by short perp (+0.01% / day)
                funding_earned = equity * abs(pos_weight) * 0.0001
                equity += funding_earned
                total_funding_earned += funding_earned

        equity_curve.append({
            'date': dt,
            'equity': equity,
            'pos_weight': pos_weight,
            'close': cl
        })

    df_res = pd.DataFrame(equity_curve)
    
    # Calculate performance metrics
    total_days = (df_bt['date'].max() - df_bt['date'].min()).days + 1
    total_return = (equity - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
    
    df_res['peak'] = df_res['equity'].cummax()
    df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak']
    max_drawdown = df_res['drawdown'].min()
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    df_res['daily_ret'] = df_res['equity'].pct_change().fillna(0.0)
    rf_daily = 0.02 / 365.0
    sharpe_ratio = (df_res['daily_ret'].mean() - rf_daily) / df_res['daily_ret'].std() * np.sqrt(365)
    
    print("=== BACKTEST RESULTS ===")
    print(f"Period: {df_bt['date'].min().strftime('%Y-%m-%d')} to {df_bt['date'].max().strftime('%Y-%m-%d')} ({total_days} days)")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${equity:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Trades: {trade_count}")
    print(f"Stop Outs Count: {stop_outs_count}")
    print(f"Funding Fee Earned (Short Perp): ${total_funding_earned:,.2f}")
    print(f"Total Trading Fees Paid: ${total_trade_fees:,.2f}")
    
    # Quarterly Breakdown
    df_res['year_quarter'] = df_res['date'].dt.to_period('Q')
    quarters = df_res['year_quarter'].unique()
    
    print("\n=== QUARTERLY BREAKDOWN ===")
    q_data = []
    for q in quarters:
        df_q = df_res[df_res['year_quarter'] == q]
        q_start_eq = df_bt_prev = df_res[df_res['date'] < df_q['date'].min()]['equity'].iloc[-1] if len(df_res[df_res['date'] < df_q['date'].min()]) > 0 else initial_capital
        q_end_eq = df_q['equity'].iloc[-1]
        q_ret = (q_end_eq - q_start_eq) / q_start_eq
        q_peak = df_q['equity'].cummax()
        q_mdd = ((df_q['equity'] - q_peak) / q_peak).min()
        
        q_data.append({
            'Quarter': str(q),
            'Start Equity': f"${q_start_eq:,.2f}",
            'End Equity': f"${q_end_eq:,.2f}",
            'Return (%)': f"{q_ret*100:.2f}%",
            'Max Drawdown (%)': f"{q_mdd*100:.2f}%"
        })
    
    df_q_summary = pd.DataFrame(q_data)
    print(df_q_summary.to_string(index=False))

run_backtest()
