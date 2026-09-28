# BTC ETF + SOPR 量化策略与 Telegram 机器人 (Strategy-etf)

基于美国现货比特币 ETF 净流入/流出数据与 SOPR 链上筹码成本基准的量化交易策略及每日 Telegram 自动化机器人。

## Telegram 机器人与 GitHub Actions 部署

本仓库包含了通过 GitHub Actions 每日自动抓取数据、直接读取 Hyperliquid 链上钱包真实持仓，并推送持仓/调仓通知到 Telegram 手机端的完整代码：

### 必填 Secret 配置
在 GitHub 仓库中进入 **Settings -> Secrets and variables -> Actions**，添加以下密钥：
1. `TG_BOT_TOKEN`: 你的 Telegram Bot Token (来自 @BotFather)
2. `TG_CHAT_ID`: 你的 Telegram Chat ID (来自 @userinfobot)
3. `HYPERLIQUID_WALLET` *(可选)*: 你的 Hyperliquid 链上钱包地址 (0x 开头)

### 核心文件
- `bot.py`: 每日运行抓取数据、评估持仓与发送 TG 消息的主程序
- `trade_state.json`: 持仓状态持久化文件
- `.github/workflows/daily_bot.yml`: 每日 UTC 00:30 (北京时间 08:30) 自动触发运行的 Cron 配置
- `btc_etf_sopr_option_b.py`: 回测源码
