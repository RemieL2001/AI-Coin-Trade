import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="트레이딩 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
<style>
body, .reportview-container .main {
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: #f8f9fa;
    color: #222;
}
table.styled-table {
    border-collapse: collapse;
    margin: 0 auto;
    font-size: 0.9em;
    min-width: 800px;
    background-color: #ffffff;
    box-shadow: 0 4px 25px 3px rgba(80,98,168,0.10);
}
table.styled-table thead tr {
    background-color: #4682b4;
    color: #ffffff;
    text-align: center;
    font-size: 1.0em;
}
table.styled-table th,
table.styled-table td {
    padding: 12px 15px;
    text-align: center;
    border-bottom: 1px solid #e1e9f0;
}
table.styled-table tbody tr:nth-of-type(even) {
    background-color: #eaf1fb;
}
table.styled-table tbody tr:hover {
    background-color: #b2c9df;
    color: #222;
}
td .buy { color: #228b22; font-weight: bold; }
td .sell { color: #b22222; font-weight: bold; }
@media (max-width: 900px) {
  table.styled-table {
    display: block;
    overflow-x: scroll;
  }
}
</style>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.header("조회 설정")
lookback = st.sidebar.slider(
    "최근 몇 건을 조회할까요?",
    min_value=10, max_value=200, value=50, step=10
)

@st.cache_data
def get_trade_statistics(db_path: str, lookback: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT decision, price FROM trading_base ORDER BY trade_id DESC LIMIT ?", (lookback,)
    )
    rows = cur.fetchall()
    conn.close()

    profit_list, last_buy = [], None
    for dec, prc in reversed(rows):
        if dec == "BUY":
            last_buy = prc
        elif dec == "SELL" and last_buy:
            profit_list.append((prc - last_buy) / last_buy)
            last_buy = None
    trade_count = len(profit_list)
    win_rate = sum(1 for p in profit_list if p > 0) / trade_count * 100 if trade_count else 0
    avg_profit = sum(profit_list) / trade_count * 100 if trade_count else 0
    return round(win_rate, 2), round(avg_profit, 2)

win_rate, avg_profit = get_trade_statistics("trading_history.db", lookback)
c1, c2 = st.columns(2)
c1.metric("🔋 승률(%)", f"{win_rate}")
c2.metric("💰 평균 수익률(%)", f"{avg_profit}")

@st.cache_data
def load_dashboard_data(db_path: str, limit: int):
    sql = """
    SELECT
      datetime(tb.timestamp, '+9 hours') AS '시간',
      tb.ticker       AS '티커',
      tb.decision     AS '실제결정',
      tb.price        AS '실제가격',
      tb.balance      AS '잔고',
      aa.analysis_result AS 'AI결정',
      aa.confidence_score AS '확신도',
      aa.reason         AS 'AI이유',
      md.current_price  AS '시장가격',
      md.volume         AS '시장거래량'
    FROM trading_base tb
    LEFT JOIN ai_analysis aa
      ON tb.ticker = aa.ticker AND DATE(aa.timestamp)=DATE(tb.timestamp)
    LEFT JOIN market_data md
      ON tb.ticker = md.ticker AND DATE(md.timestamp)=DATE(tb.timestamp)
    ORDER BY tb.timestamp DESC
    LIMIT ?
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(sql, conn, params=(limit,))
    conn.close()
    return df

df = load_dashboard_data("trading_history.db", lookback)

# 스타일 지정
def highlight_class(val):
    if val == "BUY":
        return '<span class="buy">BUY</span>'
    elif val == "SELL":
        return '<span class="sell">SELL</span>'
    else:
        return val

df['실제결정'] = df['실제결정'].apply(highlight_class)
df['AI결정']   = df['AI결정'].apply(highlight_class)
df['확신도']   = df['확신도'].apply(lambda v: f"{v:,.0%}" if pd.notna(v) else '')

html = df.to_html(index=False, classes="styled-table", border=0, escape=False)

st.markdown(html, unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; margin-top:15px;">'
    '<a href="/api/trades" target="_blank">🔗 JSON API 보기</a>'
    '</div>',
    unsafe_allow_html=True
)