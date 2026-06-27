# ST亚振摘帽事件监测仪表盘

## 文件
- `st_yazhen_event_monitor_app.py`：Streamlit 主程序
- `st_yazhen_requirements.txt`：依赖包

## 本地运行
```bash
pip install -r st_yazhen_requirements.txt
streamlit run st_yazhen_event_monitor_app.py
```

## Streamlit Cloud 部署
1. 新建 GitHub 仓库。
2. 上传 `st_yazhen_event_monitor_app.py` 和 `st_yazhen_requirements.txt`。
3. 将 `st_yazhen_requirements.txt` 改名为 `requirements.txt`。
4. Streamlit Cloud 选择主文件 `st_yazhen_event_monitor_app.py` 部署。

## 功能
- 实时/准实时行情
- 日线趋势、成交额、换手率、RSI、MACD、ATR、VaR/CVaR
- 分钟 VWAP 承接判断
- T+1 与 T+0/可日内工具两套信号
- Monte Carlo 价格情景模拟

## 重要限制
- 公开数据源可能延迟或中断。
- A股通过港股通/北向渠道一般仍不是 T+0；是否可做空取决于交易所名单和券商权限。
- 程序不构成投资建议。
