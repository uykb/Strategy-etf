import urllib.request
import json
import os
import re
import sys
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Force UTF-8 encoding for stdout printing
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
STATE_FILE = "trade_state.json"

def send_telegram_message(message: str):
    """发送 Telegram 消息"""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Warning: TG_BOT_TOKEN or TG_CHAT_ID not set. Outputting message locally:\n")
        print(message)
        return
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("Telegram message sent successfully.")
        else:
            print(f"Failed to send Telegram message: {res.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def fetch_btc_data():
    """获取 BTC/USDT 最新日线数据 (支持多节点备用，防止 451 区域限制)"""
    urls = [
        'https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=200', # 官方无地域限制节点
        'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=200',
        'https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1d&limit=200'
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    data = None
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    print(f"Successfully fetched BTC data from {url}")
                    break
        except Exception as e:
            print(f"Endpoint {url} failed: {e}, trying fallback...")
            
    if not data:
        raise RuntimeError("All BTC API endpoints failed. Unable to fetch BTC daily price data.")

    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)

def fetch_farside_etf_data():
    """获取 Farside 比特币 ETF 每日净流入数据"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    res = requests.get('https://farside.co.uk/bitcoin-etf-flow-all-data/', headers=headers, timeout=15)
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

def load_state():
    """读取持仓状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "pos_state": 0,
        "pos_weight": 0.0,
        "entry_price": 0.0,
        "sl_price": 0.0,
        "tp_price": 0.0,
        "entry_date": "",
        "equity": 1000.0
    }

def save_state(state):
    """保存持仓状态"""
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

def run_daily_bot():
    print("=== 开始运行 BTC ETF + SOPR 每日策略机器人 ===")
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()
    
    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].fillna(0.0)
    
    # 指标计算
    prev_close = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr_14'] = df['tr'].rolling(14).mean()
    cost_basis = df['close'].ewm(span=28, adjust=False).mean()
    df['sopr_28ma'] = (df['close'] / cost_basis).rolling(28).mean()
    df['vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)
    
    row_today = df.iloc[-1]
    row_prev = df.iloc[-2]
    
    dt_today = row_today['date'].strftime('%Y-%m-%d')
    close_today = row_today['close']
    high_today = row_today['high']
    low_today = row_today['low']
    open_today = row_today['open']
    
    s_flow = row_prev['etf_flow']
    s_sopr = row_prev['sopr_28ma']
    s_atr = row_prev['atr_14']
    s_vol20 = row_prev['vol_20']
    vol_75th = df['vol_20'].tail(60).quantile(0.75)
    
    state = load_state()
    pos_state = state["pos_state"]
    equity = state["equity"]
    
    # 场景 1: 当前有持仓 -> 检查 1.5x SL 或 3.5x TP
    if pos_state != 0:
        entry_p = state["entry_price"]
        sl_p = state["sl_price"]
        tp_p = state["tp_price"]
        pos_w = state["pos_weight"]
        exited = False
        exit_reason = ""
        exec_price = 0.0
        
        if pos_state > 0: # 多头持仓
            if low_today <= sl_p:
                exited = True
                exit_reason = "🚨 **止损触发 (-1.5x ATR)**"
                exec_price = min(open_today, sl_p)
                pnl_pct = (exec_price - entry_p) / entry_p
            elif high_today >= tp_p:
                exited = True
                exit_reason = "🎉 **止盈触发 (+3.5x ATR)**"
                exec_price = max(open_today, tp_p)
                pnl_pct = (exec_price - entry_p) / entry_p

            if exited:
                equity += equity * pos_w * pnl_pct - (equity * abs(pos_w) * 0.0010)
                msg = f"""{exit_reason}
📅 **日期**: {dt_today}
📈 **动作**: 多头平仓 (LONG Exit)
💵 **建仓价**: ${entry_p:,.2f}
🎯 **平仓价**: ${exec_price:,.2f}
📊 **单笔收益**: {pnl_pct*100:+.2f}%
💰 **最新账户总净值**: ${equity:,.2f}
"""
                send_telegram_message(msg)
                state["pos_state"] = 0
                state["pos_weight"] = 0.0
                state["entry_price"] = 0.0
                state["sl_price"] = 0.0
                state["tp_price"] = 0.0
                state["equity"] = equity
                save_state(state)
                return

        elif pos_state < 0: # 空头持仓
            if high_today >= sl_p:
                exited = True
                exit_reason = "🚨 **止损触发 (-1.5x ATR)**"
                exec_price = max(open_today, sl_p)
                pnl_pct = (entry_p - exec_price) / entry_p
            elif low_today <= tp_p:
                exited = True
                exit_reason = "🎉 **止盈触发 (+3.5x ATR)**"
                exec_price = min(open_today, tp_p)
                pnl_pct = (entry_p - exec_price) / entry_p

            if exited:
                equity += equity * abs(pos_w) * pnl_pct - (equity * abs(pos_w) * 0.0005)
                msg = f"""{exit_reason}
📅 **日期**: {dt_today}
📉 **动作**: 空头平仓 (SHORT Exit)
💵 **建仓价**: ${entry_p:,.2f}
🎯 **平仓价**: ${exec_price:,.2f}
📊 **单笔收益**: {pnl_pct*100:+.2f}%
💰 **最新账户总净值**: ${equity:,.2f}
"""
                send_telegram_message(msg)
                state["pos_state"] = 0
                state["pos_weight"] = 0.0
                state["entry_price"] = 0.0
                state["sl_price"] = 0.0
                state["tp_price"] = 0.0
                state["equity"] = equity
                save_state(state)
                return

        if not exited:
            dist_sl = abs(close_today - sl_p) / close_today * 100
            dist_tp = abs(close_today - tp_p) / close_today * 100
            pos_str = "多头 (LONG)" if pos_state > 0 else "空头 (SHORT)"
            msg = f"""📊 **BTC ETF 策略日常持仓监控**
📅 **日期**: {dt_today}
🔒 **当前状态**: 锁定持有 {pos_str}
💵 **建仓价格**: ${entry_p:,.2f}
当前 BTC 价格: ${close_today:,.2f}
🛑 **止损价**: ${sl_p:,.2f} (距止损 {dist_sl:.2f}%)
🎯 **止盈价**: ${tp_p:,.2f} (距止盈 {dist_tp:.2f}%)
💰 **当前本金净值**: ${equity:,.2f}
"""
            send_telegram_message(msg)
            return

    # 场景 2: 当前无持仓 (FLAT) -> 检查开仓
    if pos_state == 0:
        if s_flow > 0 and s_sopr > 1.0:
            pos_w = 0.40 if s_vol20 > vol_75th else 0.70
            entry_p = open_today
            sl_p = entry_p - 1.5 * s_atr
            tp_p = entry_p + 3.5 * s_atr
            
            state["pos_state"] = 1
            state["pos_weight"] = pos_w
            state["entry_price"] = entry_p
            state["sl_price"] = sl_p
            state["tp_price"] = tp_p
            state["entry_date"] = dt_today
            save_state(state)
            
            msg = f"""🚀 **BTC ETF + SOPR 开仓通知 (LONG)**
📅 **日期**: {dt_today}
📈 **动作**: 买入现货做多 (仓位 {pos_w*100:.0f}%)
💵 **开仓价格**: ${entry_p:,.2f}
🛑 **挂止损价 (-1.5x ATR)**: ${sl_p:,.2f}
🎯 **挂止盈价 (+3.5x ATR)**: ${tp_p:,.2f}
📊 **前日信号**: ETF流入 ${s_flow:.1f}M | SOPR={s_sopr:.4f}
"""
            send_telegram_message(msg)

        elif s_flow < 0 and s_sopr < 1.0:
            pos_w = -0.40 if s_vol20 > vol_75th else -0.45
            entry_p = open_today
            sl_p = entry_p + 1.5 * s_atr
            tp_p = entry_p - 3.5 * s_atr
            
            state["pos_state"] = -1
            state["pos_weight"] = pos_w
            state["entry_price"] = entry_p
            state["sl_price"] = sl_p
            state["tp_price"] = tp_p
            state["entry_date"] = dt_today
            save_state(state)
            
            msg = f"""📉 **BTC ETF + SOPR 开仓通知 (SHORT)**
📅 **日期**: {dt_today}
📉 **动作**: 2倍永续做空 (名义仓位 {pos_w*100:.0f}%)
💵 **开仓价格**: ${entry_p:,.2f}
🛑 **挂止损价 (-1.5x ATR)**: ${sl_p:,.2f}
🎯 **挂止盈价 (+3.5x ATR)**: ${tp_p:,.2f}
📊 **前日信号**: ETF流出 ${s_flow:.1f}M | SOPR={s_sopr:.4f}
"""
            send_telegram_message(msg)

        else:
            msg = f"""💤 **BTC ETF 策略今日观望 (FLAT)**
📅 **日期**: {dt_today}
📊 **前日信号**: ETF资金流 ${s_flow:.1f}M | SOPR 28MA = {s_sopr:.4f}
当前无持仓，等待下一个“ETF + SOPR”共振信号。
"""
            send_telegram_message(msg)

if __name__ == '__main__':
    run_daily_bot()
