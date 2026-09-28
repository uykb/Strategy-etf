# BTC ETF + SOPR 量化策略与 Telegram 机器人 (Strategy-etf)

基于美国现货比特币 ETF 净流入/流出数据与 SOPR 链上筹码成本基准的量化交易策略及每日 Telegram 自动化机器人。

## 策略说明 (方案 B)

- **多头信号**: `ETF_Flow > 0` 且 `SOPR_28MA > 1.0` -> 开仓现货多头 (70%)
- **空头信号**: `ETF_Flow < 0` 且 `SOPR_28MA < 1.0` -> 开仓 2倍永续空头 (-45% 名义敞口)
- **风控与挂单**:
  - **止损 (SL)**: `1.5x ATR(14)`
  - **止盈 (TP)**: `3.5x ATR(14)`
  - **盈亏比**: `2.33 : 1`
  - **前视偏差处理**: 严格执行 `shift(1D)`
  - **状态机**: 有持仓时锁定等待止盈/止损触发，触及平仓后恢复无持仓状态重新等待信号。

## 回测表现 (2025.01 - 2026.09)

- **初始本金**: $1,000.00
- **最终净值**: $1,355.10 (**+35.51%** 累计收益率)
- **2026 今年收益 (YTD)**: **+17.82%** (同期 BTC 现货 -4.92%)
- **最大回撤 (MDD)**: -14.45% (同期 BTC 现货 -52.97%)
- **胜率**: 44.44% (16 胜 / 20 负)

## Telegram 机器人与 GitHub Actions 部署

本仓库包含了通过 GitHub Actions 每日自动抓取数据并推送持仓/调仓通知到 Telegram 手机端的完整代码：

### 必填 Secret 配置
在 GitHub 仓库中进入 **Settings -> Secrets and variables -> Actions**，添加以下密钥：
1. `TG_BOT_TOKEN`: 你的 Telegram Bot Token (来自 @BotFather)
2. `TG_CHAT_ID`: 你的 Telegram Chat ID (来自 @userinfobot)

### 核心文件
- `bot.py`: 每日运行抓取数据、评估持仓与发送 TG 消息的主程序
- `trade_state.json`: 持仓状态持久化文件
- `.github/workflows/daily_bot.yml`: 每日 UTC 00:30 (北京时间 08:30) 自动触发运行的 Cron 配置
- `btc_etf_sopr_option_b.py`: 回测源码
