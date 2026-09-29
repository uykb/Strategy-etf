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

# Clean and format Telegram credentials
raw_token = os.environ.get("TG_BOT_TOKEN", "").strip()
if raw_token.lower().startswith("bot"):
    TG_BOT_TOKEN = raw_token[3:].strip()
else:
    TG_BOT_TOKEN = raw_token

TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()
HYPERLIQUID_WALLET = os.environ.get("HYPERLIQUID_WALLET", "").strip()
STATE_FILE = "trade_state.json"

def send_telegram_message(message: str):
    """发送 Telegram 消息"""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Warning: TG_BOT_TOKEN or TG_CHAT_ID not set in environment. Printing message locally:\n")
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
            print("Telegram message sent successfully!")
        else:
            print(f"Failed to send Telegram message. HTTP Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def fetch_btc_data():
    """获取 BTC/USDT 最新日线数据 (全量 fallback 链)"""
    urls = [
        'https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api-gcp.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api1.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api2.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api3.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api4.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300',
        'https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1d&limit=300'
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
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
            print(f"Endpoint {url} failed ({e}), trying next fallback...")
            
    if not data:
        raise RuntimeError("All Binance API fallback nodes failed. Unable to fetch BTC daily price data.")

    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)

def fetch_farside_etf_data():
    """获取 Farside 比特币 ETF 每日净流入数据 (多表格扫描 + 全拟真 Chrome 指纹)"""
    urls = [
        'https://farside.co.uk/bitcoin-etf-flow-all-data/',
        'https://farside.co.uk/btc/',
        'https://farside.co.uk/us-btc-etf-flow-data/'
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1'
    }
    
    parsed = []
    
    # 1. 优先使用 curl_cffi
    try:
        from curl_cffi import requests as c_requests
        for url in urls:
            try:
                res = c_requests.get(url, headers=headers, impersonate='chrome120', timeout=15)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    tables = soup.find_all('table')
                    for table in tables:
                        for r in table.find_all('tr'):
                            cols = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                            if len(cols) >= 2 and re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', cols[0]):
                                val_col = cols[-1] if cols[-1] not in ['', '-'] else (cols[-2] if len(cols) > 2 else '0')
                                val_str = val_col.replace('$', '').replace(',', '').strip()
                                try:
                                    val = -float(val_str[1:-1]) if val_str.startswith('(') and val_str.endswith(')') else (float(val_str) if val_str not in ['-', '', 'NaN'] else 0.0)
                                    parsed.append({'date_str': cols[0], 'etf_flow': val})
                                except ValueError:
                                    pass
                    if len(parsed) > 0:
                        print(f"Successfully fetched & parsed {len(parsed)} ETF flow rows via curl_cffi from {url}")
                        break
            except Exception as e:
                print(f"curl_cffi fetch {url} failed: {e}")
    except ImportError:
        print("curl_cffi not installed, falling back to standard requests.")

    # 2. 降级备用 requests
    if len(parsed) == 0:
        for url in urls:
            try:
                res = requests.get(url, headers=headers, timeout=15)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for table in soup.find_all('table'):
                        for r in table.find_all('tr'):
                            cols = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                            if len(cols) >= 2 and re.search(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', cols[0]):
                                val_col = cols[-1] if cols[-1] not in ['', '-'] else (cols[-2] if len(cols) > 2 else '0')
                                val_str = val_col.replace('$', '').replace(',', '').strip()
                                try:
                                    val = -float(val_str[1:-1]) if val_str.startswith('(') and val_str.endswith(')') else (float(val_str) if val_str not in ['-', '', 'NaN'] else 0.0)
                                    parsed.append({'date_str': cols[0], 'etf_flow': val})
                                except ValueError:
                                    pass
                    if len(parsed) > 0:
                        print(f"Successfully fetched & parsed {len(parsed)} ETF flow rows via requests from {url}")
                        break
            except Exception as e:
                print(f"requests fetch {url} failed: {e}")

    ETF_CACHE_FILE = "etf_cache.json"

    if len(parsed) == 0:
        print("Warning: Farside parsing yielded 0 rows. Attempting to restore from local etf_cache.json...")
        if os.path.exists(ETF_CACHE_FILE):
            try:
                with open(ETF_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                df_cached = pd.DataFrame(cache_data)
                df_cached['date'] = pd.to_datetime(df_cached['date'])
                print(f"Successfully restored {len(df_cached)} ETF flow rows from local cache.")
                return df_cached.sort_values('date')[['date', 'etf_flow']].drop_duplicates(subset=['date']).reset_index(drop=True)
            except Exception as e:
                print(f"Error loading ETF cache: {e}")
        print("Warning: No local ETF cache available. Returning empty fallback dataframe.")
        return pd.DataFrame(columns=['date', 'etf_flow'])

    df_etf = pd.DataFrame(parsed)
    df_etf['date'] = pd.to_datetime(df_etf['date_str'], format='%d %b %Y', errors='coerce')
    df_clean = df_etf.dropna(subset=['date']).sort_values('date')[['date', 'etf_flow']].drop_duplicates(subset=['date']).reset_index(drop=True)
    
    # Save cache to disk for future fallbacks
    try:
        cache_data = df_clean.to_dict(orient='records')
        with open(ETF_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, default=str, indent=2)
    except Exception as e:
        print(f"Failed to write ETF cache: {e}")
        
    return df_clean

def fetch_hyperliquid_position(wallet_address: str):
    """直接调用 Hyperliquid L1 API 查询指定钱包地址的 BTC 链上真实持仓"""
    if not wallet_address:
        return None
        
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "clearinghouseState",
        "user": wallet_address
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            asset_positions = data.get("assetPositions", [])
            for ap in asset_positions:
                pos = ap.get("position", {})
                coin = pos.get("coin", "")
                if coin == "BTC":
                    szi = float(pos.get("szi", 0.0))
                    entry_px = float(pos.get("entryPx", 0.0))
                    unrealized_pnl = float(pos.get("unrealizedPnl", 0.0))
                    return {
                        "has_position": szi != 0.0,
                        "szi": szi,
                        "side": "LONG" if szi > 0 else ("SHORT" if szi < 0 else "FLAT"),
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized_pnl
                    }
            return {"has_position": False, "side": "FLAT", "szi": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0}
        else:
            print(f"Hyperliquid API Error: {res.status_code}, {res.text}")
            return None
    except Exception as e:
        print(f"Exception fetching Hyperliquid position: {e}")
        return None

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

def calculate_ahr999(df_btc):
    """
    计算 AHR999 囤币指标 (仅供推送参考，不参与交易决策)
    使用 2026 重拟参数: slope = 5.64, intercept = 16.33
    公式: AHR999 = (现价 / 200日几何均值) * (现价 / 指数拟合估值)
    Genesis = 2009-01-03
    """
    if df_btc is None or len(df_btc) < 200:
        return None
        
    try:
        row_today = df_btc.iloc[-1]
        close_today = float(row_today['close'])
        date_val = row_today['date']
        date_today = date_val.date() if hasattr(date_val, 'date') else date_val

        # 200 日几何均值 (Geometric Mean)
        close_200 = df_btc['close'].iloc[-200:].astype(float)
        gm200 = float(np.exp(np.log(close_200).mean()))

        # 币龄天数 (比特币创世块 2009-01-03)
        genesis = datetime.date(2009, 1, 3)
        age_days = (date_today - genesis).days
        if age_days <= 0:
            return None

        log_age = np.log10(age_days)

        # 2026 新重拟参数 (slope = 5.64, intercept = 16.33)
        fit_2026 = 10 ** (5.64 * log_age - 16.33)
        ahr_2026 = (close_today / gm200) * (close_today / fit_2026)

        if ahr_2026 < 0.45:
            zone_str = "抄底区间 (<0.45)"
        elif ahr_2026 < 1.2:
            zone_str = "定投区间 (0.45~1.2)"
        else:
            zone_str = "止投区间 (≥1.2)"

        return {
            "ahr_2026": ahr_2026,
            "zone_2026": zone_str,
            "gm200": gm200
        }
    except Exception as e:
        print(f"Error calculating AHR999: {e}")
        return None

def format_ahr999_message(ahr_info):
    if not ahr_info:
        return ""
    return (
        f"💡  **AHR999**: `{ahr_info['ahr_2026']:.4f}` | {ahr_info['zone_2026']}\n"
        f"💰 **定投成本**: `${ahr_info['gm200']:,.2f}`"
    )

def run_daily_bot():
    print("=== 开始运行 BTC ETF + SOPR 每日策略机器人 ===")
    df_btc = fetch_btc_data()
    df_etf = fetch_farside_etf_data()
    
    # 计算 AHR999 囤币指标 (仅供推送，不参与策略决策)
    ahr_info = calculate_ahr999(df_btc)
    ahr_str = format_ahr999_message(ahr_info)
    
    # 避免周末与发布延迟导致 fillna(0.0) 抹零，改用 ffill()
    df = pd.merge(df_btc, df_etf, on='date', how='left')
    df['etf_flow'] = df['etf_flow'].ffill().fillna(0.0)
    
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
    
    # 提取美股最新交易日的具体数值与日期
    if not df_etf.empty:
        latest_etf_row = df_etf.iloc[-1]
        latest_etf_date = latest_etf_row['date'].strftime('%m-%d')
        latest_etf_val = latest_etf_row['etf_flow']
        etf_display_str = f"${latest_etf_val:+.1f}M ({latest_etf_date})"
    else:
        etf_display_str = f"${row_prev['etf_flow']:+.1f}M (数据暂缺)"
        
    s_flow = row_prev['etf_flow']
    s_sopr = row_prev['sopr_28ma']
    s_atr = row_prev['atr_14']
    s_vol20 = row_prev['vol_20']
    vol_75th = df['vol_20'].tail(60).quantile(0.75)
    
    # 优先使用 Hyperliquid 链上真实钱包持仓
    hl_pos = None
    if HYPERLIQUID_WALLET:
        print(f"正在查询 Hyperliquid 钱包地址真实持仓: {HYPERLIQUID_WALLET[:6]}...{HYPERLIQUID_WALLET[-4:]}")
        hl_pos = fetch_hyperliquid_position(HYPERLIQUID_WALLET)
        if hl_pos:
            print(f"Hyperliquid 链上真实持仓状态: {hl_pos['side']} | 开仓价: ${hl_pos['entry_price']:,.2f} | 数量: {hl_pos['szi']} BTC")

    state = load_state()
    
    # 确定生效状态
    if hl_pos:
        pos_state = 1 if hl_pos['side'] == 'LONG' else (-1 if hl_pos['side'] == 'SHORT' else 0)
        entry_p = hl_pos['entry_price'] if pos_state != 0 else 0.0
    else:
        pos_state = state["pos_state"]
        entry_p = state["entry_price"]

    # === 获取当日新信号 ===
    high_vol = s_vol20 > vol_75th
    if s_sopr > 1.005: sopr_state = "bull"
    elif s_sopr < 0.995: sopr_state = "bear"
    else: sopr_state = "neutral"
    
    if s_flow == 0:
        sig_type, action_str, pos_rec_str = "ZERO_FLOW", "维持现有仓位不变", "不变"
    elif s_flow > 0:
        if sopr_state == "bull":
            sig_type, action_str, pos_rec_str = "LONG_STRONG", "做多现货/永续", ("40% (高波动率压缩)" if high_vol else "60%-80% (建议70%)")
        elif sopr_state == "neutral":
            sig_type, action_str, pos_rec_str = "LONG_WEAK", "轻仓做多", ("30%-40% (高波动率压缩)" if high_vol else "30%-50%")
        else:
            sig_type, action_str, pos_rec_str = "CONFLICT", "观望/极轻仓", "≤20%"
    else:
        if sopr_state == "bear":
            sig_type, action_str, pos_rec_str = "SHORT_STRONG", "做空 (2倍永续)", ("名义40% 保证金20% (高波动率压缩)" if high_vol else "名义40%-50% 保证金20%-25%")
        elif sopr_state == "neutral":
            sig_type, action_str, pos_rec_str = "SHORT_WEAK", "轻仓做空 (2倍永续)", ("名义30%-40% 保证金15%-20% (高波动率压缩)" if high_vol else "名义30%-50% 保证金15%-25%")
        else:
            sig_type, action_str, pos_rec_str = "CONFLICT", "观望/极轻仓", "≤20%"

    # -------------------------------------------------------------
    # 场景 1: 当前有持仓 (LONG 或 SHORT) -> 检查止损/止盈/信号反转/矛盾
    # -------------------------------------------------------------
    if pos_state != 0:
        is_long = pos_state > 0
        sl_p = entry_p - 1.5 * s_atr if is_long else entry_p + 1.5 * s_atr
        tp_p = entry_p + 3.5 * s_atr if is_long else entry_p - 3.5 * s_atr
        
        exited = False
        exit_reason = ""
        
        # 1. ATR 追踪止损/止盈 (触发立即平仓)
        if is_long:
            if low_today <= sl_p: exited, exit_reason = True, "🚨 **Hyperliquid 多头已触发 1.5x ATR 止损！建议立即平仓**"
            elif high_today >= tp_p: exited, exit_reason = True, "🎉 **Hyperliquid 多头已触发 3.5x ATR 止盈！建议立即平仓**"
        else:
            if high_today >= sl_p: exited, exit_reason = True, "🚨 **Hyperliquid 空头已触发 1.5x ATR 止损！建议立即平仓**"
            elif low_today <= tp_p: exited, exit_reason = True, "🎉 **Hyperliquid 空头已触发 3.5x ATR 止盈！建议立即平仓**"
            
        if exited:
            msg = f"""{exit_reason}
📅 **日       期**: {dt_today}
⚡ **链上平台**: Hyperliquid
📈 **方       向**: {"多头 (LONG)" if is_long else "空头 (SHORT)"}
💵 **开仓参考价**: ${entry_p:,.2f}
当前 BTC 价格: ${close_today:,.2f}
🛑 **止损触发价**: ${sl_p:,.2f}
🎯 **止盈触发价**: ${tp_p:,.2f}
{ahr_str}"""
            send_telegram_message(msg)
            if not hl_pos: state["pos_state"] = 0; save_state(state)
            return

        # 2. 未触碰止盈止损，检查信号状态 (反转/矛盾/维持)
        pos_str = "多头 (LONG)" if is_long else "空头 (SHORT)"
        hl_info_str = f"⚡ **Hyperliquid 真实持仓**: {hl_pos['szi']} BTC (未实现盈亏: ${hl_pos['unrealized_pnl']:+,.2f})\n" if hl_pos else ""
        dist_sl = abs(close_today - sl_p) / close_today * 100
        dist_tp = abs(close_today - tp_p) / close_today * 100
        
        if sig_type == "ZERO_FLOW":
            notice = "✅ 零流入日：不主动平仓，维持现有仓位与止损。"
        elif sig_type == "CONFLICT":
            notice = "⚠️ **信号矛盾**：建议减仓至 ≤20% 或清仓！"
        elif (is_long and "SHORT" in sig_type) or (not is_long and "LONG" in sig_type):
            notice = f"🚨 **信号发生反转**：建议先平掉旧{pos_str}，并反手 {action_str}！"
        else:
            notice = f"✅ 信号同向 ({sig_type})：继续持有，注意止损。"
            
        msg = f"""📊 **Hyperliquid 链上持仓日常监控**
📅 **日       期**: {dt_today}
🔒 **当前状态**: 持有 {pos_str}
{hl_info_str}💵 **建仓参考价**: ${entry_p:,.2f}
当前 BTC 价格: ${close_today:,.2f}
🛑 **设置止损 (-1.5x ATR)**: ${sl_p:,.2f} (距止损 {dist_sl:.2f}%)
🎯 **设置止盈 (+3.5x ATR)**: ${tp_p:,.2f} (距止盈 {dist_tp:.2f}%)
📝 **操作建议**: {notice}
{ahr_str}"""
        send_telegram_message(msg)
        return

    # -------------------------------------------------------------
    # 场景 2: 当前无持仓 (FLAT) -> 检查开仓信号
    # -------------------------------------------------------------
    if pos_state == 0:
        entry_p = open_today
        if "LONG" in sig_type:
            sl_p = entry_p - 1.5 * s_atr
            tp_p = entry_p + 3.5 * s_atr
            if not hl_pos: state["pos_state"] = 1; state["entry_price"] = entry_p; state["sl_price"] = sl_p; state["tp_price"] = tp_p; save_state(state)
            
            msg = f"""🚀 **BTC ETF + SOPR 开仓信号 ({sig_type})**
📅 **日       期**: {dt_today}
⚡ **建议行动**: **{action_str}**
📈 **推荐仓位**: {pos_rec_str}
💵 **建仓参考价**: ${entry_p:,.2f}
🛑 **设置止损 (-1.5x ATR)**: ${sl_p:,.2f}
🎯 **设置止盈 (+3.5x ATR)**: ${tp_p:,.2f}
📊 **前日信号**: ETF {etf_display_str} | SOPR={s_sopr:.4f}
{ahr_str}"""
            send_telegram_message(msg)
            
        elif "SHORT" in sig_type:
            sl_p = entry_p + 1.5 * s_atr
            tp_p = entry_p - 3.5 * s_atr
            if not hl_pos: state["pos_state"] = -1; state["entry_price"] = entry_p; state["sl_price"] = sl_p; state["tp_price"] = tp_p; save_state(state)
            
            msg = f"""📉 **BTC ETF + SOPR 开仓信号 ({sig_type})**
📅 **日       期**: {dt_today}
⚡ **建议行动**: **{action_str}**
📉 **推荐仓位**: {pos_rec_str}
💵 **建仓参考价**: ${entry_p:,.2f}
🛑 **设置止损 (-1.5x ATR)**: ${sl_p:,.2f}
🎯 **设置止盈 (+3.5x ATR)**: ${tp_p:,.2f}
📊 **前日信号**: ETF {etf_display_str} | SOPR={s_sopr:.4f}
{ahr_str}"""
            send_telegram_message(msg)
            
        else:
            msg = f"""💤 **Hyperliquid 策略今日观望 (FLAT)**
📅 **日       期**: {dt_today}
📊 **最新美股交易日**: ETF {etf_display_str} | SOPR 28MA = {s_sopr:.4f}
当前无持仓，等待下一个明确开仓信号。
📝 **操作建议**: {action_str} ({pos_rec_str})
{ahr_str}"""
            send_telegram_message(msg)

if __name__ == '__main__':
    run_daily_bot()
