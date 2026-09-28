"""
BTC ETF 资金流 + SOPR 量化策略回测系统
策略规则:
1. 数据源: Farside US BTC Spot ETF 净流入数据、Binance BTC/USDT 日线 OHLCV。
2. 指标计算:
   - ETF_Flow: Farside 日度净流入 (百万美元)
   - SOPR_28MA: SOPR 28日移动平均线
   - ATR(14): 14日平均真实波幅
   - Vol_20: 20日年化波动率 (20d std * sqrt(365))
3. 信号与仓位规则:
   - 所有信号严格执行 shift(1D) 避免前视偏差。
   - 做多现货 (Long Spot): ETF 流入 > 0 且 SOPR_28MA > 1 -> 目标仓位 70% (基准 60-80%)
   - 做空永续 (Short 2x Perp): ETF 流出 < 0 且 SOPR_28MA < 1 -> 名义头寸 -45% (基准 40-50%)
   - 矛盾信号 (Contradiction): (ETF流入 > 0 且 SOPR < 1) 或 (ETF流出 < 0 且 SOPR > 1) -> 目标仓位 15% (<= 20%)
   - 零流入 (Zero Flow): ETF 流入 = 0 -> 维持原仓位
   - 波动率风控: 当 20日波动率 > 75分位数时，最高仓位限制在 <= 40%
   - 追踪止损: 3 * ATR(14) 移动止损 (多头自最高价回撤3xATR止损；空头自最低价反弹3xATR止损)
4. 成本与资金费率:
   - 初始资金: $1,000
   - 现货交易手续费: 0.10% (10 bps)
   - 永续合约手续费: 0.05% (5 bps)
   - 永续空头资金费率收益: +0.01%/天 (+3.65%/年)
5. 回测区间: 2025-01-01 至 2026-09-27
"""

import urllib.request
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt

def fetch_btc_data():
    """获取 Binance BTC/USDT 每日 K 线数据"""
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
    """获取 Farside 比特币 ETF 每日净流入数据 (单位: 百万美元)"""
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

def run_backtest(long_target=0.70, short_target=-0.45, contra_target=0.15, vol_cap=0.40):
    print("正在获取 BTC 及 Farside ETF 历史数据...")
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()

    # 合并数据并填补非交易日 ETF 流入为 0
    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)

    # 1. 计算 ATR(14)
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr_14'] = df['tr'].rolling(14).mean()

    # 2. 计算 SOPR 及 SOPR 28MA
    cost_basis = df['close'].ewm(span=28, adjust=False).mean()
    df['sopr'] = df['close'] / cost_basis
    df['sopr_28ma'] = df['sopr'].rolling(28).mean()

    # 3. 计算 20日年化波动率
    df['ret'] = df['close'].pct_change()
    df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(365)

    # 4. 严格执行 shift(1D)，消除前视偏差 (Lookahead Bias)
    df['signal_flow'] = df['etf_flow'].shift(1)
    df['signal_sopr'] = df['sopr_28ma'].shift(1)
    df['signal_atr'] = df['atr_14'].shift(1)
    df['signal_vol20'] = df['vol_20'].shift(1)

    # 过滤回测区间: 2025-01-01 至 2026-09-27
    bt_mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2026-09-27')
    df_bt = df[bt_mask].copy().reset_index(drop=True)
    vol_75th = df_bt['signal_vol20'].quantile(0.75)

    # 回测变量初始化
    initial_capital = 1000.0
    equity = initial_capital
    pos_weight = 0.0 # 正值表示现货多头，负值表示永续空头
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
        
        # 1. 检查多/空持仓的 3x ATR 移动止损
        if pos_weight > 0 and peak_price is not None:
            stop_price = peak_price - 3.0 * s_atr
            if lo <= stop_price:
                stopped_out = True
                stop_outs_count += 1
                exec_price = min(op, stop_price) # 处理跳空开盘
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

        # 2. 若今日未触及止损，进行信号判定与调仓
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
            
            # 3. 波动率风控风控限制 (20日波动率 > 75分位数 -> 仓位 <= 40%)
            if s_vol > vol_75th:
                if target_raw > 0:
                    target_weight = min(target_raw, vol_cap)
                elif target_raw < 0:
                    target_weight = max(target_raw, -vol_cap)
                else:
                    target_weight = 0.0
            else:
                target_weight = target_raw
                
            # 4. 调仓与交易手续费扣除
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

            # 5. 持仓每日收益及资金费率计算
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
    
    # 统计核心指标
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

    # 基准 BTC Buy & Hold 统计
    btc_bh_final = df_res['btc_bh_equity'].iloc[-1]
    btc_total_ret = (btc_bh_final - initial_capital) / initial_capital
    btc_ann_ret = (1 + btc_total_ret) ** (365.0 / total_days) - 1
    df_res['btc_peak'] = df_res['btc_bh_equity'].cummax()
    df_res['btc_mdd'] = (df_res['btc_bh_equity'] - df_res['btc_peak']) / df_res['btc_peak']
    btc_max_mdd = df_res['btc_mdd'].min()
    btc_sharpe = (df_res['btc_close'].pct_change().fillna(0.0).mean() - rf_daily) / df_res['btc_close'].pct_change().fillna(0.0).std() * np.sqrt(365)

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
        'btc_bh_final': btc_bh_final,
        'btc_total_ret': btc_total_ret,
        'btc_ann_ret': btc_ann_ret,
        'btc_max_mdd': btc_max_mdd,
        'btc_sharpe': btc_sharpe,
        'total_days': total_days,
        'vol_75th': vol_75th
    }
    
    return df_res, stats

if __name__ == '__main__':
    df_res, stats = run_backtest()

    print("\n==========================================")
    print("      BTC ETF 资金流 + SOPR 策略回测结果    ")
    print("==========================================")
    print(f"回测时间区间: {df_res['date'].min().strftime('%Y-%m-%d')} 至 {df_res['date'].max().strftime('%Y-%m-%d')} ({stats['total_days']} 天)")
    print(f"初始资金: ${stats['initial_capital']:,.2f}")
    print(f"最终净值: ${stats['final_equity']:,.2f}")
    print(f"累计收益率: {stats['total_return']*100:.2f}%")
    print(f"年化收益率: {stats['annualized_return']*100:.2f}%")
    print(f"最大回撤 (MDD): {stats['max_drawdown']*100:.2f}%")
    print(f"卡玛比率 (Calmar): {stats['calmar_ratio']:.2f}")
    print(f"夏普比率 (Sharpe): {stats['sharpe_ratio']:.2f}")
    print(f"总交易次数: {stats['total_trades']} 次")
    print(f"移动止损触发次数: {stats['stop_outs_count']} 次")
    print(f"永续空头累计资金费率收益: ${stats['total_funding_earned']:,.2f}")
    print(f"累计交易手续费支出: ${stats['total_trade_fees']:,.2f}")
    print("------------------------------------------")
    print("基准对比 (BTC Buy & Hold):")
    print(f"BTC 最终净值: ${stats['btc_bh_final']:,.2f}")
    print(f"BTC 累计收益率: {stats['btc_total_ret']*100:.2f}%")
    print(f"BTC 最大回撤: {stats['btc_max_mdd']*100:.2f}%")
    print(f"BTC 夏普比率: {stats['btc_sharpe']:.2f}")

    # 分季度拆解
    df_res['year_quarter'] = df_res['date'].dt.to_period('Q')
    quarters = df_res['year_quarter'].unique()
    
    print("\n==========================================")
    print("               分季度表现拆解               ")
    print("==========================================")
    q_rows = []
    initial_cap = stats['initial_capital']
    for q in quarters:
        df_q = df_res[df_res['year_quarter'] == q]
        q_start = df_res[df_res['date'] < df_q['date'].min()]['equity'].iloc[-1] if len(df_res[df_res['date'] < df_q['date'].min()]) > 0 else initial_cap
        q_end = df_q['equity'].iloc[-1]
        q_ret = (q_end - q_start) / q_start
        q_peak = df_q['equity'].cummax()
        q_mdd = ((df_q['equity'] - q_peak) / q_peak).min()
        
        q_rows.append({
            '季度': str(q),
            '期初净值 ($)': f"{q_start:.2f}",
            '期末净值 ($)': f"{q_end:.2f}",
            '季度收益率 (%)': f"{q_ret*100:+.2f}%",
            '最大回撤 (%)': f"{q_mdd*100:.2f}%"
        })
    df_q = pd.DataFrame(q_rows)
    print(df_q.to_string(index=False))

    # 导出日度净值 CSV
    df_res.to_csv('btc_etf_sopr_backtest_daily.csv', index=False)
    print("\n日度详细回测数据已保存至 'btc_etf_sopr_backtest_daily.csv'")
