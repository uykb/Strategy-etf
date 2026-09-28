"""
BTC ETF 资金流 + SOPR 挂单止盈止损策略 (方案 B - 今年爆发型)
配置参数:
1. 信号: 纯粹 ETF_Flow > 0 且 SOPR_28MA > 1.0 (开多 70% 现货)
         ETF_Flow < 0 且 SOPR_28MA < 1.0 (开空 -45% 永续)
2. 信号处理: 严格 shift(1D) 消除前视偏差
3. 止损 (SL): 1.5x ATR(14)
4. 止盈 (TP): 3.5x ATR(14)
5. 盈亏比 (RR): 2.33 : 1
6. 状态机: 有持仓时拒绝新信号，锁定等待 1.5x SL 或 3.5x TP 触发后平仓重置
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

def run_option_b_backtest(sl_mult=1.5, tp_mult=3.5):
    print("正在拉取 BTC 现货及 Farside ETF 最新数据...")
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()

    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)

    # 1. 指标计算
    prev_close = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr_14'] = df['tr'].rolling(14).mean()
    cost_basis = df['close'].ewm(span=28, adjust=False).mean()
    df['sopr_28ma'] = (df['close'] / cost_basis).rolling(28).mean()
    df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)

    # 2. 严格执行 shift(1D)，消除前视偏差 (Lookahead Bias)
    df['signal_flow'] = df['etf_flow'].shift(1)
    df['signal_sopr'] = df['sopr_28ma'].shift(1)
    df['signal_atr'] = df['atr_14'].shift(1)
    df['signal_vol20'] = df['vol_20'].shift(1)

    # 过滤回测区间: 2025-01-01 至 2026-09-27
    bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
    df_bt = df[bt_mask].copy().reset_index(drop=True)
    vol_75th = df_bt['signal_vol20'].quantile(0.75)

    initial_capital = 1000.0
    equity = initial_capital
    pos_state = 0 # 0: FLAT, +1: LONG, -1: SHORT
    pos_weight = 0.0

    entry_price, sl_price, tp_price = None, None, None
    win_count, loss_count, total_fees, funding_earned = 0, 0, 0.0, 0.0

    equity_curve = []
    trade_log = []

    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        dt, op, hi, lo, cl = row['date'], row['open'], row['high'], row['low'], row['close']
        c_prev = df_bt.iloc[i-1]['close'] if i > 0 else op
        s_flow, s_sopr, s_atr, s_vol20 = row['signal_flow'], row['signal_sopr'], row['signal_atr'], row['signal_vol20']

        # 状态 1: 有持仓 -> 锁定等待 1.5x SL 或 3.5x TP
        if pos_state != 0:
            exited = False
            exec_price = None
            exit_reason = None

            if pos_state > 0: # 多头持仓
                if lo <= sl_price: # 触及 1.5x ATR 止损
                    exited = True
                    exit_reason = '止损 (-1.5x ATR)'
                    exec_price = min(op, sl_price)
                    loss_count += 1
                elif hi >= tp_price: # 触及 3.5x ATR 止盈
                    exited = True
                    exit_reason = '止盈 (+3.5x ATR)'
                    exec_price = max(op, tp_price)
                    win_count += 1

                if exited:
                    equity += equity * pos_weight * ((exec_price - c_prev) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0010
                    equity -= fee
                    total_fees += fee
                    
                    trade_log.append({
                        'exit_date': dt.strftime('%Y-%m-%d'),
                        'type': '多头 (LONG)',
                        'entry_price': f"${entry_price:,.2f}",
                        'exit_price': f"${exec_price:,.2f}",
                        'sl_price': f"${sl_price:,.2f}",
                        'tp_price': f"${tp_price:,.2f}",
                        'reason': exit_reason,
                        'equity': f"${equity:,.2f}"
                    })
                    pos_state, pos_weight, entry_price = 0, 0.0, None

            elif pos_state < 0: # 空头持仓
                if hi >= sl_price: # 触及 1.5x ATR 止损
                    exited = True
                    exit_reason = '止损 (-1.5x ATR)'
                    exec_price = max(op, sl_price)
                    loss_count += 1
                elif lo <= tp_price: # 触及 3.5x ATR 止盈
                    exited = True
                    exit_reason = '止盈 (+3.5x ATR)'
                    exec_price = min(op, tp_price)
                    win_count += 1

                if exited:
                    equity += equity * abs(pos_weight) * ((c_prev - exec_price) / c_prev)
                    fee = equity * abs(pos_weight) * 0.0005
                    equity -= fee
                    total_fees += fee
                    
                    trade_log.append({
                        'exit_date': dt.strftime('%Y-%m-%d'),
                        'type': '空头 (SHORT)',
                        'entry_price': f"${entry_price:,.2f}",
                        'exit_price': f"${exec_price:,.2f}",
                        'sl_price': f"${sl_price:,.2f}",
                        'tp_price': f"${tp_price:,.2f}",
                        'reason': exit_reason,
                        'equity': f"${equity:,.2f}"
                    })
                    pos_state, pos_weight, entry_price = 0, 0.0, None

        # 状态 2: 无持仓 -> 触发信号才开仓
        if pos_state == 0:
            if s_flow > 0 and s_sopr > 1.0: # 触发做多信号
                pos_state = 1
                pos_weight = 0.40 if s_vol20 > vol_75th else 0.70
                entry_price = op
                sl_price = entry_price - sl_mult * s_atr
                tp_price = entry_price + tp_mult * s_atr
                fee = equity * pos_weight * 0.0010
                equity -= fee
                total_fees += fee

            elif s_flow < 0 and s_sopr < 1.0: # 触发做空信号
                pos_state = -1
                pos_weight = -0.40 if s_vol20 > vol_75th else -0.45
                entry_price = op
                sl_price = entry_price + sl_mult * s_atr
                tp_price = entry_price - tp_mult * s_atr
                fee = equity * abs(pos_weight) * 0.0005
                equity -= fee
                total_fees += fee

        # 每日持仓盈亏及资金费收益计算
        if pos_state > 0:
            equity += equity * pos_weight * ((cl - c_prev) / c_prev)
        elif pos_state < 0:
            equity += equity * abs(pos_weight) * ((c_prev - cl) / c_prev)
            f_earn = equity * abs(pos_weight) * 0.0001
            equity += f_earn
            funding_earned += f_earn

        btc_start = df_bt.iloc[0]['close']
        btc_bh_equity = initial_capital * (cl / btc_start)

        equity_curve.append({
            'date': dt,
            'equity': equity,
            'pos_state': pos_state,
            'pos_weight': pos_weight,
            'btc_close': cl,
            'btc_bh_equity': btc_bh_equity
        })

    df_res = pd.DataFrame(equity_curve)

    # 计算全局核心指标
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

    # 提取 2026 今年 (YTD) 专项指标
    df_2026 = df_res[df_res['date'] >= '2026-01-01'].reset_index(drop=True)
    st_2026 = df_res[df_res['date'] < '2026-01-01']['equity'].iloc[-1]
    ed_2026 = df_2026['equity'].iloc[-1]
    ret_2026 = (ed_2026 - st_2026) / st_2026
    df_2026['peak'] = df_2026['equity'].cummax()
    mdd_2026 = ((df_2026['equity'] - df_2026['peak']) / df_2026['peak']).min()

    btc_start_2026 = df_2026['btc_close'].iloc[0]
    btc_end_2026 = df_2026['btc_close'].iloc[-1]
    btc_ret_2026 = (btc_end_2026 - btc_start_2026) / btc_start_2026

    print("\n==========================================")
    print("  BTC ETF + SOPR 方案 B (1.5x SL / 3.5x TP)   ")
    print("==========================================")
    print(f"回测区间: {df_bt['date'].min().strftime('%Y-%m-%d')} 至 {df_bt['date'].max().strftime('%Y-%m-%d')} ({total_days} 天)")
    print(f"初始本金: ${initial_capital:,.2f}")
    print(f"最终净值: ${equity:,.2f}")
    print(f"累计收益率: {total_return*100:+.2f}%")
    print(f"年化收益率: {annualized_return*100:+.2f}%")
    print(f"最大回撤 (MDD): {max_drawdown*100:.2f}%")
    print(f"卡玛比率 (Calmar): {calmar_ratio:.2f}")
    print(f"夏普比率 (Sharpe): {sharpe_ratio:.2f}")
    print(f"胜率 (Win Rate): {win_rate:.2f}% ({win_count} 胜 / {loss_count} 负)")
    print(f"总交易笔数: {win_count + loss_count} 笔")
    print(f"永续空头累计资金费收益: ${funding_earned:,.2f}")
    print(f"累计交易手续费支出: ${total_fees:,.2f}")
    print("------------------------------------------")
    print("2026 今年表现 (2026.01.01 - 2026.09.27):")
    print(f"2026 年初净值: ${st_2026:.2f} -> 最新净值: ${ed_2026:.2f}")
    print(f"2026 收益率 (YTD): {ret_2026*100:+.2f}% (同期 BTC 现货: {btc_ret_2026*100:+.2f}%)")
    print(f"2026 最大回撤 (MDD): {mdd_2026*100:.2f}%")

    # 分季度拆解
    df_res['year_quarter'] = df_res['date'].dt.to_period('Q')
    quarters = df_res['year_quarter'].unique()
    print("\n==========================================")
    print("               分季度表现拆解               ")
    print("==========================================")
    q_data = []
    for q in quarters:
        df_q = df_res[df_res['year_quarter'] == q]
        q_start = df_res[df_res['date'] < df_q['date'].min()]['equity'].iloc[-1] if len(df_res[df_res['date'] < df_q['date'].min()]) > 0 else initial_capital
        q_end = df_q['equity'].iloc[-1]
        q_ret = (q_end - q_start) / q_start
        q_peak = df_q['equity'].cummax()
        q_mdd = ((df_q['equity'] - q_peak) / q_peak).min()
        
        q_data.append({
            '季度 Period': str(q),
            '期初净值 ($)': f"{q_start:.2f}",
            '期末净值 ($)': f"{q_end:.2f}",
            '季度收益 (%)': f"{q_ret*100:+.2f}%",
            '最大回撤 (%)': f"{q_mdd*100:.2f}%"
        })
    df_q_table = pd.DataFrame(q_data)
    print(df_q_table.to_string(index=False))

    # 导出日度净值 CSV
    df_res.to_csv('btc_etf_sopr_option_b_daily.csv', index=False)
    print("\n日度详细回测数据已保存至 'btc_etf_sopr_option_b_daily.csv'")

if __name__ == '__main__':
    run_option_b_backtest(sl_mult=1.5, tp_mult=3.5)
