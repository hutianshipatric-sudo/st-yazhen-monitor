# -*- coding: utf-8 -*-
"""
ST亚振（603389）事件驱动实时监测仪表盘
用途：监测摘帽/摘ST事件前后，辅助判断 T+1 与 T+0/可日内工具 两种交易模式下的风险、动量、波动和入场信号。

运行：
    pip install -r requirements.txt
    streamlit run st_yazhen_event_monitor_app.py

重要声明：本程序只做行情与风险量化监控，不构成买卖建议。A股/港股通/券商可做空规则请以券商和交易所为准。
"""

import math
import time
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# AkShare 是核心数据源；如果部署环境拉取不稳定，页面会给出错误提示。
try:
    import akshare as ak
except Exception as e:
    ak = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# =========================
# 基础配置
# =========================
st.set_page_config(
    page_title="ST亚振摘帽事件监测",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYMBOL_DEFAULT = "603389"
NAME_DEFAULT = "ST亚振"

# =========================
# 工具函数
# =========================
def _to_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "")
            if x.strip() in ["-", "--", ""]:
                return default
        return float(x)
    except Exception:
        return default


def _safe_pct(x):
    if pd.isna(x) or np.isinf(x):
        return "--"
    return f"{x:.2f}%"


def _safe_num(x, digits=2):
    if pd.isna(x) or np.isinf(x):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_money(x):
    x = _to_float(x)
    if pd.isna(x):
        return "--"
    if abs(x) >= 1e8:
        return f"{x/1e8:.2f} 亿"
    if abs(x) >= 1e4:
        return f"{x/1e4:.2f} 万"
    return f"{x:.0f}"


def _clamp(v, lo=0, hi=100):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return np.nan


def percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)


def next_weekday(d: date, weekday: int) -> date:
    """weekday: Monday=0 ... Sunday=6"""
    days = (weekday - d.weekday()) % 7
    if days == 0:
        days = 7
    return d + timedelta(days=days)


# =========================
# 数据获取
# =========================
@st.cache_data(ttl=20)
def fetch_spot(symbol: str) -> Tuple[pd.DataFrame, Dict]:
    if ak is None:
        return pd.DataFrame(), {"error": "AkShare 未安装。请先 pip install akshare"}
    try:
        df = ak.stock_zh_a_spot_em()
        # 常见列名：代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 今开, 昨收, 量比, 换手率, 市盈率-动态, 市净率, 总市值, 流通市值, 涨速, 5分钟涨跌, 60日涨跌幅, 年初至今涨跌幅
        row = df[df["代码"].astype(str).str.zfill(6) == symbol.zfill(6)]
        meta = {}
        if row.empty:
            meta["error"] = f"实时列表中未找到代码 {symbol}"
        else:
            meta = row.iloc[0].to_dict()
        return df, meta
    except Exception as e:
        return pd.DataFrame(), {"error": f"实时行情获取失败：{e}"}


@st.cache_data(ttl=120)
def fetch_daily(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol.zfill(6),
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"])
        num_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("日期").reset_index(drop=True)
    except Exception as e:
        st.warning(f"日线数据获取失败：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=25)
def fetch_minute(symbol: str, period: str, days: int) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol.zfill(6),
            start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            period=period,
            adjust="",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # 常见列名：时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 均价
        df["时间"] = pd.to_datetime(df["时间"])
        for c in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "均价"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("时间").reset_index(drop=True)
    except Exception as e:
        st.info(f"分钟数据暂时不可用：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_news(symbol: str) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    # AkShare 个股新闻接口有版本差异，这里做容错。
    for fn_name in ["stock_news_em"]:
        try:
            fn = getattr(ak, fn_name)
            df = fn(symbol=symbol.zfill(6))
            if df is not None and not df.empty:
                return df.head(20)
        except Exception:
            pass
    return pd.DataFrame()


# =========================
# 指标计算
# =========================
def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["收盘"]
    high = out["最高"]
    low = out["最低"]
    volume = out["成交量"] if "成交量" in out else pd.Series(index=out.index, dtype=float)
    amount = out["成交额"] if "成交额" in out else pd.Series(index=out.index, dtype=float)
    turnover = out["换手率"] if "换手率" in out else pd.Series(index=out.index, dtype=float)

    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close / close.shift(1))
    for n in [3, 5, 10, 20, 60]:
        out[f"MA{n}"] = close.rolling(n).mean()
        out[f"RET{n}"] = close / close.shift(n) - 1
        out[f"VOL{n}"] = out["ret"].rolling(n).std() * np.sqrt(252)
        out[f"AMT_MA{n}"] = amount.rolling(n).mean()
        out[f"TURN_MA{n}"] = turnover.rolling(n).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    out["DEA"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = 2 * (out["DIF"] - out["DEA"])

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["ATR_PCT"] = out["ATR14"] / close

    # Bollinger
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB_MID"] = mid
    out["BB_UPPER"] = mid + 2 * std
    out["BB_LOWER"] = mid - 2 * std
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / mid

    # 量能/换手分位
    out["AMT_PCTL_60"] = out["成交额"].rolling(60).apply(lambda x: percentile_rank(pd.Series(x[:-1]), x[-1]) if len(x) > 10 else np.nan, raw=False)
    if "换手率" in out.columns:
        out["TURN_PCTL_60"] = out["换手率"].rolling(60).apply(lambda x: percentile_rank(pd.Series(x[:-1]), x[-1]) if len(x) > 10 else np.nan, raw=False)

    return out


def add_minute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["收盘"]
    amount = out["成交额"] if "成交额" in out else pd.Series(index=out.index, dtype=float)
    vol = out["成交量"] if "成交量" in out else pd.Series(index=out.index, dtype=float)
    out["ret"] = close.pct_change()
    out["MA5"] = close.rolling(5).mean()
    out["MA20"] = close.rolling(20).mean()
    out["VWAP"] = (out["成交额"].cumsum() / out["成交量"].replace(0, np.nan).cumsum()) if "成交额" in out and "成交量" in out else np.nan
    out["AMT_MA20"] = amount.rolling(20).mean()
    out["VOL_MA20"] = vol.rolling(20).mean()
    out["minute_volatility"] = out["ret"].rolling(30).std() * math.sqrt(240)
    return out


def calculate_scores(daily: pd.DataFrame, minute: pd.DataFrame, spot: Dict, event_date: date, limit_pct_now: float, limit_pct_after: float) -> Dict:
    """输出 Long/Short/Risk/Event 四类评分与解释。"""
    scores = {
        "long_score": np.nan,
        "short_score": np.nan,
        "risk_score": np.nan,
        "event_score": np.nan,
        "entry_signal_t0": "数据不足",
        "entry_signal_t1": "数据不足",
        "reasons_long": [],
        "reasons_short": [],
        "risk_flags": [],
    }
    if daily.empty or len(daily) < 30:
        return scores

    last = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) >= 2 else last
    price = _to_float(spot.get("最新价", np.nan), default=np.nan)
    if pd.isna(price) or price <= 0:
        price = _to_float(last.get("收盘", np.nan), default=np.nan)

    close = _to_float(last.get("收盘"))
    pct_chg = _to_float(spot.get("涨跌幅", last.get("涨跌幅", np.nan)))
    amount = _to_float(spot.get("成交额", last.get("成交额", np.nan)))
    turnover = _to_float(spot.get("换手率", last.get("换手率", np.nan)))
    vol_ratio = _to_float(spot.get("量比", np.nan))

    ma5 = _to_float(last.get("MA5")); ma10 = _to_float(last.get("MA10")); ma20 = _to_float(last.get("MA20"))
    rsi = _to_float(last.get("RSI14")); macd = _to_float(last.get("MACD_HIST"))
    ret5 = _to_float(last.get("RET5")) * 100
    ret10 = _to_float(last.get("RET10")) * 100
    atr_pct = _to_float(last.get("ATR_PCT")) * 100
    vol20 = _to_float(last.get("VOL20")) * 100
    amt_pct = percentile_rank(daily["成交额"].tail(60), amount)
    turn_pct = percentile_rank(daily["换手率"].tail(60), turnover) if "换手率" in daily else np.nan

    # 事件距离：越接近摘帽，事件分越高，但兑现风险也高。
    days_to_event = (event_date - datetime.now().date()).days
    event_score = 50
    if days_to_event in [0, 1, 2]:
        event_score += 25
    elif 3 <= days_to_event <= 5:
        event_score += 15
    elif days_to_event < 0:
        event_score -= 10
    event_score += 10 if abs(pct_chg) > 2 else 0
    event_score = _clamp(event_score)

    # Long score：趋势 + 承接 + 量能 + 事件催化，过热扣分。
    long_score = 50
    if price > ma5 > ma10:
        long_score += 10; scores["reasons_long"].append("价格站上 MA5/MA10，短线趋势偏强")
    if price > ma20:
        long_score += 5; scores["reasons_long"].append("价格位于 MA20 上方，中短期趋势未破")
    if macd > 0:
        long_score += 7; scores["reasons_long"].append("MACD 柱为正，动量仍在")
    if 50 <= rsi <= 75:
        long_score += 8; scores["reasons_long"].append("RSI 处于强势但未极端过热区")
    elif rsi > 85:
        long_score -= 10; scores["risk_flags"].append("RSI 极端过热，追高风险上升")
    if not pd.isna(amt_pct) and amt_pct >= 70:
        long_score += 8; scores["reasons_long"].append("成交额处于近 60 日高分位，资金活跃")
    if not pd.isna(vol_ratio) and vol_ratio >= 1.2:
        long_score += 5; scores["reasons_long"].append("量比放大，事件资金正在参与")
    if 0 < pct_chg < limit_pct_now * 0.85:
        long_score += 5; scores["reasons_long"].append("上涨但未逼近涨停，承接相对健康")
    if pct_chg >= limit_pct_now * 0.9:
        long_score -= 7; scores["risk_flags"].append("接近涨停/高开极限，T+1 买入后隔夜风险较高")
    if ret10 > 35:
        long_score -= 8; scores["risk_flags"].append("10 日涨幅过大，利好兑现风险上升")
    long_score += (event_score - 50) * 0.25

    # Short score：高开兑现、跌破均价、过热、放量滞涨。
    short_score = 40
    if rsi > 80:
        short_score += 10; scores["reasons_short"].append("RSI 高位，存在兑现压力")
    if ret5 > 20 or ret10 > 35:
        short_score += 10; scores["reasons_short"].append("短期累计涨幅过大，容易出现利好兑现")
    if not pd.isna(amt_pct) and amt_pct >= 85 and pct_chg < 2:
        short_score += 8; scores["reasons_short"].append("成交额高分位但价格不强，疑似放量滞涨")
    if macd < 0:
        short_score += 8; scores["reasons_short"].append("MACD 柱转弱，动量衰减")
    if price < ma5:
        short_score += 7; scores["reasons_short"].append("跌破 MA5，短线趋势破坏")

    # 风险评分：波动 + 过热 + T+1锁定 + 临近事件。
    risk_score = 35
    if atr_pct > 5:
        risk_score += 15; scores["risk_flags"].append("ATR% 较高，单日波动容错低")
    elif atr_pct > 3:
        risk_score += 8; scores["risk_flags"].append("ATR% 偏高，需要缩小仓位")
    if vol20 > 80:
        risk_score += 12; scores["risk_flags"].append("20日年化波动率过高")
    if ret10 > 35:
        risk_score += 15
    if days_to_event in [0, 1, 2]:
        risk_score += 10; scores["risk_flags"].append("事件临近，波动和兑现风险同步升高")
    if pct_chg >= limit_pct_now * 0.9:
        risk_score += 10
    risk_score = _clamp(risk_score)

    # 分钟级承接修正
    intraday_note = []
    if minute is not None and not minute.empty and len(minute) >= 30:
        mlast = minute.iloc[-1]
        mprice = _to_float(mlast.get("收盘"), price)
        vwap = _to_float(mlast.get("VWAP"), np.nan)
        mma5 = _to_float(mlast.get("MA5"), np.nan)
        mma20 = _to_float(mlast.get("MA20"), np.nan)
        intraday_ret = (mprice / _to_float(minute.iloc[0].get("开盘", mprice)) - 1) * 100
        if not pd.isna(vwap) and mprice > vwap:
            long_score += 7; intraday_note.append("分钟价格站上 VWAP，日内承接强")
        elif not pd.isna(vwap) and mprice < vwap:
            long_score -= 8; short_score += 8; intraday_note.append("分钟价格跌破 VWAP，日内承接弱")
        if not pd.isna(mma5) and not pd.isna(mma20) and mprice > mma5 > mma20:
            long_score += 5; intraday_note.append("5分钟均线强于20分钟均线，日内动量向上")
        if intraday_ret > limit_pct_now * 0.75:
            risk_score += 7; intraday_note.append("日内涨幅接近涨停区，追高风险增加")
        scores["reasons_long"].extend(intraday_note[:2])

    long_score = _clamp(long_score)
    short_score = _clamp(short_score)

    # 入场信号：不是“指令”，是客观风险/动量规则。
    now = datetime.now().time()
    # A股关键时间：09:30开盘，15:00收盘；策略里回避 09:30 前与尾盘最后几分钟。
    t_num = now.hour * 100 + now.minute

    def signal_text(mode: str) -> str:
        if t_num < 930:
            return "未开盘：只看竞价强弱，不建议提前下结论"
        if 930 <= t_num < 945:
            return "开盘前15分钟：观察承接，不急于追"
        if mode == "T0":
            if long_score >= 72 and risk_score <= 70:
                return "可观察小仓试多：需站上 VWAP/MA5，跌破立即退出"
            if short_score >= 70 and risk_score >= 60:
                return "可观察反向/做空机会：仅限有合法可做空工具，确认跌破 VWAP 后执行"
            if risk_score > 78:
                return "高风险：不追，等二次回踩或换手稳定"
            return "中性：等待方向确认"
        else:
            if long_score >= 78 and risk_score <= 62 and t_num >= 950:
                return "T+1偏稳信号：只适合小仓分批，优先等回踩不破 VWAP/MA5"
            if risk_score >= 70:
                return "T+1不友好：隔夜不可退出，建议回避追高"
            if long_score >= 68 and t_num >= 1000:
                return "T+1观察：信号尚可，但需要降低仓位并预设隔夜风险"
            return "T+1观望：等待 10:00 后承接确认"

    scores.update({
        "long_score": long_score,
        "short_score": short_score,
        "risk_score": risk_score,
        "event_score": event_score,
        "entry_signal_t0": signal_text("T0"),
        "entry_signal_t1": signal_text("T1"),
        "snapshot": {
            "price": price,
            "pct_chg": pct_chg,
            "amount": amount,
            "turnover": turnover,
            "vol_ratio": vol_ratio,
            "rsi": rsi,
            "ret5": ret5,
            "ret10": ret10,
            "atr_pct": atr_pct,
            "vol20": vol20,
            "amt_pct": amt_pct,
            "turn_pct": turn_pct,
            "days_to_event": days_to_event,
            "limit_pct_now": limit_pct_now,
            "limit_pct_after": limit_pct_after,
        }
    })
    return scores


def monte_carlo_forecast(daily: pd.DataFrame, current_price: float, horizon_days: int, n_sims: int, event_jump_mean: float, event_jump_std: float) -> pd.DataFrame:
    """历史收益重采样 + 事件跳跃项。输出模拟路径。"""
    if daily.empty or len(daily) < 60 or pd.isna(current_price) or current_price <= 0:
        return pd.DataFrame()
    rets = daily["ret"].dropna().tail(250)
    rets = rets[(rets > rets.quantile(0.01)) & (rets < rets.quantile(0.99))]
    if len(rets) < 30:
        return pd.DataFrame()
    rng = np.random.default_rng(42)
    paths = np.zeros((horizon_days + 1, n_sims))
    paths[0, :] = current_price
    for t in range(1, horizon_days + 1):
        sampled = rng.choice(rets.values, size=n_sims, replace=True)
        # 第一日叠加事件跳跃。均值/波动由用户调参。
        if t == 1:
            sampled += rng.normal(event_jump_mean / 100.0, event_jump_std / 100.0, size=n_sims)
        paths[t, :] = paths[t - 1, :] * (1 + sampled)
        paths[t, :] = np.maximum(paths[t, :], 0.01)
    return pd.DataFrame(paths)


def var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 20:
        return np.nan, np.nan
    q = r.quantile(alpha)
    cvar = r[r <= q].mean()
    return q, cvar


# =========================
# UI：侧边栏
# =========================
st.title("📈 ST亚振摘帽事件监测仪表盘")
st.caption("实时/准实时行情 + 动量 + 波动 + T+0/T+1 风险框架。信号用于辅助判断，不是确定性买卖建议。")

with st.sidebar:
    st.header("参数设置")
    symbol = st.text_input("股票代码", value=SYMBOL_DEFAULT).strip().zfill(6)
    default_event = next_weekday(datetime.now().date(), 2)  # 下一个周三
    event_dt = st.date_input("摘帽/摘ST事件日期", value=default_event)

    st.subheader("交易规则假设")
    limit_pct_now = st.number_input("当前涨跌幅限制假设（%）", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    limit_pct_after = st.number_input("摘帽后涨跌幅限制假设（%）", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    mode = st.radio("交易模式", ["T+1 A股现货", "T+0/可日内工具（仅在券商允许时）"], index=0)

    st.subheader("刷新与数据")
    refresh_sec = st.slider("自动刷新秒数", 15, 300, 30, 15)
    minute_period = st.selectbox("分钟周期", ["1", "5", "15", "30", "60"], index=1)
    minute_days = st.slider("分钟数据回看天数", 1, 10, 3)
    adjust = st.selectbox("日线复权", ["qfq", "hfq", ""], index=0, format_func=lambda x: "前复权" if x=="qfq" else ("后复权" if x=="hfq" else "不复权"))

    st.subheader("模拟预测")
    horizon_days = st.slider("预测天数", 1, 10, 3)
    n_sims = st.slider("模拟路径数", 500, 10000, 3000, 500)
    event_jump_mean = st.slider("事件跳跃均值假设（%）", -10.0, 15.0, 2.0, 0.5)
    event_jump_std = st.slider("事件跳跃波动假设（%）", 0.0, 15.0, 5.0, 0.5)

    if st_autorefresh:
        st_autorefresh(interval=refresh_sec * 1000, key="datarefresh")
    else:
        st.info("未安装 streamlit-autorefresh；页面不会自动刷新。")

# =========================
# 数据加载
# =========================
end_date = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=550)).strftime("%Y%m%d")

spot_all, spot = fetch_spot(symbol)
daily_raw = fetch_daily(symbol, start_date, end_date, adjust=adjust)
daily = add_daily_indicators(daily_raw)
minute_raw = fetch_minute(symbol, minute_period, minute_days)
minute = add_minute_indicators(minute_raw)
news = fetch_news(symbol)

if spot.get("error"):
    st.error(spot["error"])

scores = calculate_scores(daily, minute, spot, event_dt, limit_pct_now, limit_pct_after)
snap = scores.get("snapshot", {})
stock_name = spot.get("名称", NAME_DEFAULT)
current_price = snap.get("price", np.nan)

# =========================
# 顶部状态区
# =========================
cols = st.columns(6)
cols[0].metric("名称/代码", f"{stock_name} {symbol}")
cols[1].metric("最新价", _safe_num(current_price), _safe_pct(snap.get("pct_chg", np.nan)))
cols[2].metric("成交额", _fmt_money(snap.get("amount", np.nan)))
cols[3].metric("换手率", _safe_pct(snap.get("turnover", np.nan)))
cols[4].metric("量比", _safe_num(snap.get("vol_ratio", np.nan)))
cols[5].metric("距事件", f"{snap.get('days_to_event', '--')} 天")

# 信号卡片
st.subheader("1）交易信号总览")
score_cols = st.columns(4)
score_cols[0].metric("Long 动量分", _safe_num(scores.get("long_score", np.nan), 1))
score_cols[1].metric("Short/兑现分", _safe_num(scores.get("short_score", np.nan), 1))
score_cols[2].metric("总风险分", _safe_num(scores.get("risk_score", np.nan), 1))
score_cols[3].metric("事件强度分", _safe_num(scores.get("event_score", np.nan), 1))

sig1, sig2 = st.columns(2)
with sig1:
    st.info(f"**T+1 A股现货信号：** {scores.get('entry_signal_t1')}")
with sig2:
    st.warning(f"**T+0/可日内工具信号：** {scores.get('entry_signal_t0')}")

with st.expander("信号背后的逻辑", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**支持做多的因素**")
        if scores.get("reasons_long"):
            for r in scores["reasons_long"][:8]:
                st.write("- " + r)
        else:
            st.write("暂无明显做多确认。")
    with c2:
        st.markdown("**支持兑现/做空的因素**")
        if scores.get("reasons_short"):
            for r in scores["reasons_short"][:8]:
                st.write("- " + r)
        else:
            st.write("暂无明显兑现确认。")
    with c3:
        st.markdown("**风险提醒**")
        if scores.get("risk_flags"):
            for r in list(dict.fromkeys(scores["risk_flags"]))[:8]:
                st.write("- " + r)
        else:
            st.write("暂无极端风险信号，但事件票仍需小仓。")

# =========================
# 图表区
# =========================
st.subheader("2）价格、均线、成交额")
if daily.empty:
    st.warning("暂无日线数据。")
else:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=daily["日期"], open=daily["开盘"], high=daily["最高"], low=daily["最低"], close=daily["收盘"],
        name="K线"
    ))
    for ma in ["MA5", "MA10", "MA20"]:
        if ma in daily:
            fig.add_trace(go.Scatter(x=daily["日期"], y=daily[ma], mode="lines", name=ma))
    fig.add_vline(x=pd.Timestamp(event_dt), line_dash="dash", annotation_text="事件日", annotation_position="top")
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    fig_amt = px.bar(daily.tail(120), x="日期", y="成交额", title="近120日成交额")
    fig_amt.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_amt, use_container_width=True)

st.subheader("3）分钟级承接：VWAP 与短线均线")
if minute.empty:
    st.info("暂无分钟数据。开盘后或更换周期再试。")
else:
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=minute["时间"], y=minute["收盘"], mode="lines", name="分钟收盘"))
    if "VWAP" in minute:
        fig_m.add_trace(go.Scatter(x=minute["时间"], y=minute["VWAP"], mode="lines", name="VWAP"))
    for ma in ["MA5", "MA20"]:
        if ma in minute:
            fig_m.add_trace(go.Scatter(x=minute["时间"], y=minute[ma], mode="lines", name=ma))
    fig_m.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_m, use_container_width=True)

# =========================
# 风险区
# =========================
st.subheader("4）实时风险：波动、VaR/CVaR、T+1隔夜风险")
if not daily.empty and len(daily) > 60:
    last_ret = daily["ret"].dropna()
    var5, cvar5 = var_cvar(last_ret.tail(250), 0.05)
    atr_pct = snap.get("atr_pct", np.nan)
    vol20 = snap.get("vol20", np.nan)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ATR14 / 价格", _safe_pct(atr_pct))
    r2.metric("20日年化波动", _safe_pct(vol20))
    r3.metric("历史 1日 VaR 95%", _safe_pct(var5 * 100))
    r4.metric("历史 1日 CVaR 95%", _safe_pct(cvar5 * 100))

    # T+1隔夜最大不利情境
    price = current_price
    if not pd.isna(price):
        down_limit_now = price * (1 - limit_pct_now / 100)
        down_limit_after = price * (1 - limit_pct_after / 100)
        st.write(
            f"**T+1隔夜压力测试：** 若买入后不能当天卖出，下一交易日理论下跌限制按 {limit_pct_after:.1f}% 估算，"
            f"不利价格约为 **{down_limit_after:.2f}**。若仍按当前 ST 限制 {limit_pct_now:.1f}% 估算，不利价格约为 **{down_limit_now:.2f}**。"
        )
else:
    st.info("日线样本不足，无法计算 VaR/CVaR。")

# =========================
# 预测区
# =========================
st.subheader("5）价格预测：历史重采样 + 事件跳跃 Monte Carlo")
if not daily.empty and not pd.isna(current_price):
    paths = monte_carlo_forecast(daily, current_price, horizon_days, n_sims, event_jump_mean, event_jump_std)
    if not paths.empty:
        last_prices = paths.iloc[-1]
        q5, q50, q95 = last_prices.quantile([0.05, 0.5, 0.95])
        prob_up = (last_prices > current_price).mean() * 100
        prob_down_5 = (last_prices < current_price * 0.95).mean() * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("模拟上涨概率", _safe_pct(prob_up))
        m2.metric("5%分位价格", _safe_num(q5))
        m3.metric("中位数价格", _safe_num(q50))
        m4.metric("95%分位价格", _safe_num(q95))

        # 路径抽样展示
        sample_cols = paths.sample(min(80, paths.shape[1]), axis=1, random_state=1)
        fig_mc = go.Figure()
        x = list(range(0, horizon_days + 1))
        for col in sample_cols.columns:
            fig_mc.add_trace(go.Scatter(x=x, y=sample_cols[col], mode="lines", opacity=0.12, showlegend=False))
        fig_mc.add_trace(go.Scatter(x=x, y=paths.quantile(0.5, axis=1), mode="lines", name="中位数"))
        fig_mc.add_trace(go.Scatter(x=x, y=paths.quantile(0.05, axis=1), mode="lines", name="5%分位"))
        fig_mc.add_trace(go.Scatter(x=x, y=paths.quantile(0.95, axis=1), mode="lines", name="95%分位"))
        fig_mc.update_layout(height=420, xaxis_title="未来交易日", yaxis_title="价格", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_mc, use_container_width=True)
        st.caption("说明：这是基于历史收益分布与用户设定的事件跳跃项的情景模拟，不是确定性预测。")
    else:
        st.info("样本不足，暂无法模拟。")

# =========================
# 策略执行清单
# =========================
st.subheader("6）入场与风控清单")
mode_is_t0 = mode.startswith("T+0")
if mode_is_t0:
    st.markdown(
        """
**T+0/可日内工具模式（前提：券商实际允许做多/做空/日内退出）**

- **入场时间优先级：** 09:45–10:30 ＞ 13:30–14:30 ＞ 尾盘；不建议 09:30 前5分钟追。
- **做多条件：** Long分 ≥ 72、价格站上 VWAP、MA5 > MA20、风险分 ≤ 70。
- **止损条件：** 跌破 VWAP 且 3–5 分钟不能收回；或亏损超过 1个ATR的 0.5–0.7倍；或成交额放大但价格不涨。
- **兑现条件：** 接近涨停但封单不足、炸板反复、RSI>85 且跌破分钟MA5。
- **反向/做空条件：** Short分 ≥ 70、价格跌破 VWAP、放量滞涨或高开低走；仅在合法可借券/可做空工具存在时使用。
        """
    )
else:
    st.markdown(
        """
**T+1 A股现货模式**

- **入场时间优先级：** 10:00 后再判断，比 09:30–09:45 追高更稳。
- **做多条件：** Long分 ≥ 78、风险分 ≤ 62、价格回踩不破 VWAP/MA5、不是一字高开后的放量回落。
- **仓位建议框架：** 试仓 20%–30%，确认后再分批；事件票不适合一笔满仓。
- **必须回避：** 高开接近涨停后炸板、成交额高分位但价格不涨、跌破 VWAP、尾盘跳水。
- **隔夜风险：** 买入当天不能卖，第二天若低开或跌停，只能被动承受，所以信号阈值要比 T+0 更严格。
        """
    )

# =========================
# 新闻区
# =========================
st.subheader("7）新闻/公告辅助观察")
if news.empty:
    st.info("新闻接口暂时无数据。建议同时查看交易所公告、东方财富公告、券商账户公告页。")
else:
    st.dataframe(news, use_container_width=True, height=260)

st.divider()
st.caption(
    "数据源：AkShare 聚合东方财富等公开数据。公开行情可能延迟或中断；实盘下单请以券商行情、交易所公告和交易权限为准。"
)
