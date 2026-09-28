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

# Shift 1D to avoid lookahead bias
df['signal_flow'] = df['etf_flow'].shift(1)
df['signal_sopr'] = df['sopr_28ma'].shift(1)
df['signal_atr'] = df['atr_14'].shift(1)
df['signal_vol20'] = df['vol_20'].shift(1)

bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
df_bt = df[bt_mask].copy().reset_index(drop=True)
vol_75th = df_bt['signal_vol20'].quantile(0.75)

# Backtest simulation loop
initial_capital = 1000.0
equity = initial_capital
pos_state = 0 # 0: FLAT, +1: LONG, -1: SHORT
pos_weight = 0.0

entry_price = None
sl_price = None
tp_price = None

trade_count = 0
win_count = 0
loss_count = 0
stop_outs_count = 0
take_profits_count = 0

total_trade_fees = 0.0
total_funding_earned = 0.0

equity_curve = []
trade_log = []

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
    s_vol20 = row['signal_vol20']

    # State 1: Active Position - Check SL and TP
    if pos_state != 0:
        exited = False
        exec_price = None
        exit_reason = None

        if pos_state > 0: # Long Position
            # Check Stop Loss (1x ATR)
            if lo <= sl_price:
                exited = True
                exit_reason = 'Stop Loss (-1x ATR)'
                exec_price = min(op, sl_price)
                loss_count += 1
                stop_outs_count += 1
            # Check Take Profit (3x ATR)
            elif hi >= tp_price:
                exited = True
                exit_reason = 'Take Profit (+3x ATR)'
                exec_price = max(op, tp_price)
                win_count += 1
                take_profits_count += 1

            if exited:
                day_ret = (exec_price - c_prev) / c_prev
                equity += equity * pos_weight * day_ret
                # Closing fee (spot 0.1%)
                fee = equity * abs(pos_weight) * 0.0010
                equity -= fee
                total_trade_fees += fee
                
                trade_log.append({
                    'exit_date': dt,
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exec_price,
                    'reason': exit_reason,
                    'equity': equity
                })
                
                pos_state = 0
                pos_weight = 0.0
                entry_price, sl_price, tp_price = None, None, None

        elif pos_state < 0: # Short Position
            # Check Stop Loss (1x ATR)
            if hi >= sl_price:
                exited = True
                exit_reason = 'Stop Loss (-1x ATR)'
                exec_price = max(op, sl_price)
                loss_count += 1
                stop_outs_count += 1
            # Check Take Profit (3x ATR)
            elif lo <= tp_price:
                exited = True
                exit_reason = 'Take Profit (+3x ATR)'
                exec_price = min(op, tp_price)
                win_count += 1
                take_profits_count += 1

            if exited:
                day_ret = (c_prev - exec_price) / c_prev
                equity += equity * abs(pos_weight) * day_ret
                # Closing fee (perp 0.05%)
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee
                total_trade_fees += fee
                
                trade_log.append({
                    'exit_date': dt,
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exec_price,
                    'reason': exit_reason,
                    'equity': equity
                })
                
                pos_state = 0
                pos_weight = 0.0
                entry_price, sl_price, tp_price = None, None, None

    # State 2: FLAT Position - Check for Entry Signals
    if pos_state == 0:
        if s_flow > 0 and s_sopr > 1.0: # Trigger Long Spot
            pos_state = 1
            # Volatility regime cap (40% if high vol, else 70%)
            pos_weight = 0.40 if s_vol20 > vol_75th else 0.70
            entry_price = op # Enter at Open of day T
            sl_price = entry_price - 1.0 * s_atr
            tp_price = entry_price + 3.0 * s_atr
            
            # Opening fee
            fee = equity * pos_weight * 0.0010
            equity -= fee
            total_trade_fees += fee
            trade_count += 1

        elif s_flow < 0 and s_sopr < 1.0: # Trigger Short Perp
            pos_state = -1
            pos_weight = -0.40 if s_vol20 > vol_75th else -0.45
            entry_price = op # Enter at Open of day T
            sl_price = entry_price + 1.0 * s_atr
            tp_price = entry_price - 3.0 * s_atr
            
            # Opening fee
            fee = equity * abs(pos_weight) * 0.0005
            equity -= fee
            total_trade_fees += fee
            trade_count += 1

    # Daily PnL tracking for active position
    if pos_state > 0:
        day_ret = (cl - c_prev) / c_prev
        equity += equity * pos_weight * day_ret
    elif pos_state < 0:
        day_ret = (c_prev - cl) / c_prev
        equity += equity * abs(pos_weight) * day_ret
        # Funding rate earned (+0.01%/day)
        funding = equity * abs(pos_weight) * 0.0001
        equity += funding
        total_funding_earned += funding

    equity_curve.append({
        'date': dt,
        'equity': equity,
        'pos_state': pos_state,
        'pos_weight': pos_weight,
        'btc_close': cl
    })

df_res = pd.DataFrame(equity_curve)

total_days = (df_bt['date'].max() - df_bt['date'].min()).days + 1
total_return = (equity - initial_capital) / initial_capital
annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
df_res['peak'] = df_res['equity'].cummax()
df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak']
max_drawdown = df_res['drawdown'].min()
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
daily_ret = df_res['equity'].pct_change().fillna(0.0)
sharpe_ratio = (daily_ret.mean() - 0.02/365) / daily_ret.std() * np.sqrt(365)
win_rate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0

print("=== 纯粹 BTC ETF + SOPR (1x ATR 止损 / 3x ATR 止盈) 回测结果 ===")
print(f"回测时间区间: {df_bt['date'].min().strftime('%Y-%m-%d')} 至 {df_bt['date'].max().strftime('%Y-%m-%d')} ({total_days} 天)")
print(f"初始资金: ${initial_capital:,.2f}")
print(f"最终净值: ${equity:,.2f}")
print(f"累计收益率: {total_return*100:+.2f}%")
print(f"年化收益率: {annualized_return*100:+.2f}%")
print(f"最大回撤 (MDD): {max_drawdown*100:.2f}%")
print(f"卡玛比率 (Calmar): {calmar_ratio:.2f}")
print(f"夏普比率 (Sharpe): {sharpe_ratio:.2f}")
print(f"胜率 (Win Rate): {win_rate:.2f}% ({win_count} 胜 / {loss_count} 负)")
print(f"总开仓笔数: {trade_count} 笔")
print(f"止盈触发 (+3x ATR): {take_profits_count} 次")
print(f"止损触发 (-1x ATR): {stop_outs_count} 次")
print(f"永续空头累计资金费收益: ${total_funding_earned:,.2f}")
print(f"累计交易手续费支出: ${total_trade_fees:,.2f}")

# 2026 Year-to-Date Performance
df_2026 = df_res[df_res['date'] >= '2026-01-01'].reset_index(drop=True)
st_2026 = df_res[df_res['date'] < '2026-01-01']['equity'].iloc[-1]
ed_2026 = df_2026['equity'].iloc[-1]
ret_2026 = (ed_2026 - st_2026) / st_2026
mdd_2026 = ((df_2026['equity'] - df_2026['equity'].cummax()) / df_2026['equity'].cummax()).min()

print("\n--- 2026 今年表现 (2026.01.01 - 2026.09.27) ---")
print(f"2026 年初净值: ${st_2026:.2f} -> 2026 最新净值: ${ed_2026:.2f}")
print(f"2026 收益率 (YTD): {ret_2026*100:+.2f}%")
print(f"2026 最大回撤 (MDD): {mdd_2026*100:.2f}%")
