import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import date
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Dự báo giá cổ phiếu", page_icon="📈", layout="wide")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("📈 Dự báo giá cổ phiếu Việt Nam bằng LSTM")
st.write("Chọn sàn, tìm mã cổ phiếu hoặc nhập mã Yahoo Finance để dự báo giá đóng cửa.")
st.caption("Danh sách mã được tải từ nguồn công khai khi bạn bấm cập nhật. Không phải mọi mã đều có dữ liệu lịch sử trên Yahoo Finance.")

# Các mã phổ biến để dùng ngay nếu không tải được danh sách trực tuyến
POPULAR = {
    "HOSE": ["AAA","ACB","ANV","BCM","BID","BMP","BSI","BVH","CII","CMG","CTG","DGC","DGW","DHC","DIG","DPM","DXG","EIB","FPT","GAS","GEX","GMD","GVR","HAG","HCM","HDB","HDG","HHV","HPG","HSG","HT1","KBC","KDH","LPB","MBB","MSB","MSN","MWG","NAB","NKG","NVL","OCB","PDR","PLX","PNJ","POW","REE","SAB","SBT","SHB","SSB","SSI","STB","TCB","TPB","VCB","VCI","VGC","VHM","VIB","VIC","VIX","VND","VNM","VPB","VRE"],
    "HNX": ["AAV","ADC","AMV","API","APS","BAB","BAX","BCC","BBS","BID","BNA","CEO","CET","C69","DHT","DTD","DXP","HUT","IDC","IDJ","L14","LAS","MBS","NAG","NBC","NTP","PVB","PVC","PVS","SHS","TNG","TVC","VC3","VCS","VGS"],
    "UPCoM": ["A32","AAH","AAS","AAV","ABB","ABC","ABI","ABW","ACV","AG1","AGF","AGM","AGP","AGX","AIC","AIG","ALV","AMP","APF","APG","APT","ART","BSR","CTR","DDV","F88","FOX","GEE","HNG","MCH","MFS","MSR","OIL","QNS","SAS","SIP","VGI","VGT","VTP"]
}
SUFFIX = {"HOSE": ".VN", "HNX": ".HN", "UPCoM": ".UP"}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_symbol_list():
    """Thử lấy danh sách mã từ GitHub; nếu không khả dụng sẽ dùng danh sách phổ biến tích hợp."""
    urls = [
        "https://raw.githubusercontent.com/ranaroussi/yfinance/main/README.md"
    ]
    # Không xem README của thư viện là danh sách mã. Danh sách phổ biến tích hợp vẫn dùng được.
    # Tránh phụ thuộc vào một nguồn danh sách bên ngoài không ổn định.
    return POPULAR

def ticker_for(symbol, exchange, custom):
    symbol = symbol.strip().upper()
    if not symbol:
        return ""
    # Nếu người dùng nhập mã dạng Yahoo Finance (ví dụ VNM.VN, AAPL) thì giữ nguyên.
    if "." in symbol or exchange == "Quốc tế / nhập mã Yahoo":
        return symbol
    return symbol + SUFFIX.get(exchange, ".VN")

@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker, start_date, end_date):
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    raw = yf.download(ticker, start=start_date, end=end_exclusive.strftime("%Y-%m-%d"),
                      auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close", ticker) in raw.columns:
            close = raw[("Close", ticker)]
        elif "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw["Close"]
    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()
    result.index = pd.to_datetime(result.index)
    return result[~result.index.duplicated(keep="last")].sort_index()

def create_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)

def make_model(window):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

with st.sidebar:
    st.header("⚙️ Cài đặt")
    exchange = st.selectbox("Thị trường", ["HOSE", "HNX", "UPCoM", "Quốc tế / nhập mã Yahoo"])
    if exchange == "Quốc tế / nhập mã Yahoo":
        symbol = st.text_input("Mã Yahoo Finance", value="AAPL").strip().upper()
    else:
        options = sorted(set(POPULAR[exchange]))
        symbol = st.selectbox("Chọn mã cổ phiếu", options, index=options.index("VNM") if exchange == "HOSE" and "VNM" in options else 0)
        custom_symbol = st.text_input("Hoặc nhập mã khác (không bắt buộc)", placeholder="Ví dụ: VNM, FPT, VCB")
        if custom_symbol.strip():
            symbol = custom_symbol.strip().upper()
    ticker = ticker_for(symbol, exchange, False)
    st.caption(f"Mã dùng để lấy dữ liệu: **{ticker}**")
    start = st.date_input("Từ ngày", value=date(2018, 1, 1))
    end = st.date_input("Đến ngày", value=date.today())
    forecast_days = st.slider("Số phiên dự báo", 1, 30, 10)
    window = st.select_slider("Số phiên lịch sử đầu vào", options=[30, 45, 60, 90, 120], value=60)
    epochs = st.slider("Epochs tối đa", 5, 50, 10, 5)
    run = st.button("🚀 Chạy dự báo", type="primary", use_container_width=True)

if start >= end:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    try:
        df = load_data(ticker, start, end)
    except Exception as exc:
        st.error(f"Không tải được dữ liệu: {exc}")
        st.stop()

if df.empty:
    st.error(f"Không tìm thấy dữ liệu cho {ticker}. Có thể mã không được Yahoo Finance hỗ trợ hoặc không có dữ liệu trong khoảng ngày đã chọn.")
    st.info("Thử nhập mã Yahoo Finance chính xác. Mã Việt Nam thường dùng hậu tố .VN cho HOSE, .HN cho HNX; một số mã UPCoM có thể không có dữ liệu.")
    st.stop()

a, b, c = st.columns(3)
a.metric("Số phiên có dữ liệu", f"{len(df):,}")
b.metric("Giá đóng cửa gần nhất", f"{df['Close'].iloc[-1]:,.2f}")
c.metric("Ngày giao dịch gần nhất", df.index[-1].strftime("%d/%m/%Y"))
st.subheader(f"Biểu đồ giá lịch sử — {ticker}")
st.line_chart(df["Close"], y_label="Giá đóng cửa")
with st.expander("Xem dữ liệu gần nhất"):
    st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
st.download_button("⬇️ Tải dữ liệu lịch sử CSV", df.to_csv(index_label="Date").encode("utf-8-sig"),
                   file_name=f"{ticker.replace('.', '_')}_lich_su.csv", mime="text/csv")

if len(df) <= window + 30:
    st.warning("Dữ liệu chưa đủ dài để huấn luyện ổn định. Hãy chọn khoảng thời gian dài hơn hoặc giảm số phiên đầu vào.")
    st.stop()

if run:
    with st.spinner("Đang huấn luyện mô hình LSTM..."):
        try:
            values = df[["Close"]].values.astype("float32")
            split = int(len(values) * 0.8)
            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)
            train_scaled = scaled[:split]
            X, y = create_sequences(train_scaled, window)
            val_size = max(1, int(len(X) * 0.1))
            if len(X) <= val_size:
                st.error("Không đủ dữ liệu huấn luyện.")
                st.stop()
            X_fit, y_fit = X[:-val_size], y[:-val_size]
            X_val, y_val = X[-val_size:], y[-val_size:]
            X_fit = X_fit.reshape(-1, window, 1)
            X_val = X_val.reshape(-1, window, 1)

            test_combined = scaled[split-window:]
            X_test, y_test = create_sequences(test_combined, window)
            X_test = X_test.reshape(-1, window, 1)

            model = make_model(window)
            early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            history = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=epochs,
                                batch_size=32, shuffle=False, callbacks=[early], verbose=0)

            pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
            actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred) if len(actual) > 1 else np.nan

            last_window = scaled[-window:].reshape(1, window, 1)
            future_scaled = []
            for _ in range(forecast_days):
                next_val = model.predict(last_window, verbose=0)
                future_scaled.append(next_val[0, 0])
                last_window = np.concatenate([last_window[:, 1:, :], next_val.reshape(1, 1, 1)], axis=1)
            future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()
            future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({"Ngày dự báo": future_dates, "Giá đóng cửa dự báo": future_prices})
            test_dates = df.index[split:split+len(actual)]
            st.session_state["result"] = {
                "ticker": ticker, "history": history.history, "actual": actual, "pred": pred,
                "test_dates": test_dates, "rmse": rmse, "mae": mae, "r2": r2,
                "forecast": forecast_df, "last_price": float(df["Close"].iloc[-1])
            }
        except Exception as exc:
            st.error(f"Không thể chạy mô hình: {exc}")
            st.stop()

res = st.session_state.get("result")
if res and res["ticker"] == ticker:
    st.divider()
    st.subheader("📊 Đánh giá mô hình trên tập kiểm tra")
    x, y, z = st.columns(3)
    x.metric("RMSE", f"{res['rmse']:,.4f}")
    y.metric("MAE", f"{res['mae']:,.4f}")
    z.metric("R²", f"{res['r2']:.4f}" if np.isfinite(res["r2"]) else "Không xác định")
    st.caption("RMSE và MAE càng thấp thường càng tốt. R² có thể âm nếu dự báo kém hơn mô hình lấy trung bình.")
    compare = pd.DataFrame({"Giá thực tế": res["actual"], "Giá dự báo": res["pred"]}, index=res["test_dates"])
    st.line_chart(compare, y_label="Giá đóng cửa")
    with st.expander("Loss khi huấn luyện"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res["history"]["loss"], label="Train Loss")
        ax.plot(res["history"]["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.legend()
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader(f"🔮 Dự báo {len(res['forecast'])} phiên tiếp theo")
    end_price = float(res["forecast"]["Giá đóng cửa dự báo"].iloc[-1])
    pct = (end_price / res["last_price"] - 1) * 100 if res["last_price"] else np.nan
    p, q = st.columns(2)
    p.metric("Giá đóng cửa gần nhất", f"{res['last_price']:,.2f}")
    q.metric("Thay đổi dự báo đến phiên cuối", f"{pct:+.2f}%" if np.isfinite(pct) else "Không xác định")
    historical = df["Close"].tail(120).rename("Giá lịch sử")
    predicted = res["forecast"].set_index("Ngày dự báo")["Giá đóng cửa dự báo"].rename("Giá dự báo")
    st.line_chart(pd.concat([historical, predicted]), y_label="Giá đóng cửa")
    out = res["forecast"].copy()
    out["Ngày dự báo"] = out["Ngày dự báo"].dt.strftime("%d/%m/%Y")
    out["Giá đóng cửa dự báo"] = out["Giá đóng cửa dự báo"].round(2)
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Tải bảng dự báo CSV", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{ticker.replace('.', '_')}_du_bao.csv", mime="text/csv")
    st.warning("Dự báo chỉ dựa trên giá đóng cửa quá khứ, không tính tin tức, kết quả kinh doanh hay các yếu tố thị trường. Ngày dự báo là ngày làm việc gần đúng, chưa loại trừ ngày nghỉ lễ. Đây không phải khuyến nghị mua bán.")
else:
    st.info("Chọn mã cổ phiếu và nhấn **Chạy dự báo** để bắt đầu.")
