import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow is used to follow the LSTM approach in the supplied document.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


st.set_page_config(
    page_title="Stock Forecast AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 STOCK FORECAST AI")
st.caption("Dự báo giá cổ phiếu bằng mô hình LSTM — dữ liệu từ Yahoo Finance")

with st.sidebar:
    st.header("⚙️ Thiết lập")
    ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM.VN",
        help="Ví dụ Việt Nam: VNM.VN, FPT.VN, HPG.VN, VCB.VN. Mã quốc tế dùng ký hiệu Yahoo Finance tương ứng.",
    ).strip().upper()
    start_date = st.date_input("Ngày bắt đầu lấy dữ liệu", value=date(2018, 1, 1))
    end_date = st.date_input("Ngày kết thúc dữ liệu", value=date.today())
    forecast_days = st.slider("Số phiên giao dịch cần dự báo", min_value=1, max_value=30, value=10)
    epochs = st.slider("Số epochs huấn luyện tối đa", min_value=5, max_value=50, value=10, step=5)
    run_button = st.button("🚀 Tải dữ liệu và dự báo", type="primary", use_container_width=True)

st.info(
    "Lưu ý: Mô hình chỉ tham khảo dữ liệu giá đóng cửa trong quá khứ. "
    "Dự báo không đảm bảo chính xác và không phải khuyến nghị mua/bán cổ phiếu."
)

if not TF_AVAILABLE:
    st.error(
        "Chưa cài được TensorFlow. Hãy cài các thư viện trong requirements.txt "
        "bằng Python phù hợp rồi chạy lại ứng dụng."
    )
    st.stop()

if start_date >= end_date:
    st.warning("Ngày bắt đầu phải sớm hơn ngày kết thúc.")
    st.stop()

if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    with st.spinner(f"Đang tải dữ liệu {ticker}..."):
        try:
            # yfinance treats end as exclusive, so add one day to include the selected end date.
            raw = yf.download(
                ticker,
                start=start_date.isoformat(),
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if raw.empty:
                st.error("Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu hoặc khoảng thời gian.")
                st.stop()

            # yfinance may return either single-level or MultiIndex columns.
            if isinstance(raw.columns, pd.MultiIndex):
                if ("Close", ticker) in raw.columns:
                    close = raw[("Close", ticker)]
                else:
                    close_candidates = [col for col in raw.columns if col[0] == "Close"]
                    if not close_candidates:
                        st.error("Dữ liệu tải về không có cột giá đóng cửa.")
                        st.stop()
                    close = raw[close_candidates[0]]
            else:
                if "Close" not in raw.columns:
                    st.error("Dữ liệu tải về không có cột giá đóng cửa.")
                    st.stop()
                close = raw["Close"]

            df = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()
            df = df[~df.index.duplicated(keep="last")].sort_index()

            if len(df) < 100:
                st.error(
                    f"Chỉ có {len(df)} phiên dữ liệu hợp lệ. "
                    "Hãy chọn khoảng thời gian dài hơn (khuyến nghị tối thiểu 100 phiên)."
                )
                st.stop()

        except Exception as e:
            st.error(f"Không thể tải dữ liệu: {e}")
            st.stop()

    st.subheader(f"1. Dữ liệu lịch sử: {ticker}")
    latest_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    change_pct = ((latest_price / prev_price) - 1) * 100 if prev_price else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Giá đóng cửa gần nhất", f"{latest_price:,.2f}", f"{change_pct:+.2f}%")
    c2.metric("Số phiên dữ liệu", f"{len(df):,}")
    c3.metric("Ngày dữ liệu gần nhất", pd.Timestamp(df.index[-1]).strftime("%d/%m/%Y"))

    st.line_chart(df["Close"], height=300)
    with st.expander("Xem dữ liệu lịch sử"):
        st.dataframe(df.tail(100), use_container_width=True)

    # Chronological split: train first 80%, test last 20%.
    values = df[["Close"]].values.astype("float32")
    train_size = int(0.8 * len(values))
    if train_size <= 60 or len(values) - train_size < 10:
        st.error("Dữ liệu chưa đủ để chia tập huấn luyện và kiểm tra.")
        st.stop()

    train_values = values[:train_size]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    all_scaled = scaler.transform(values)

    def create_sequences(arr, window=60):
        X, y = [], []
        for i in range(window, len(arr)):
            X.append(arr[i-window:i])
            y.append(arr[i])
        return np.asarray(X), np.asarray(y)

    X_train_all, y_train_all = create_sequences(train_scaled, 60)
    val_size = max(1, int(0.1 * len(X_train_all)))
    if len(X_train_all) - val_size < 1:
        st.error("Không đủ dữ liệu sau khi tạo chuỗi huấn luyện.")
        st.stop()

    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    # Test sequences use the final 60 training observations as context.
    test_combined = np.vstack([all_scaled[train_size-60:train_size], all_scaled[train_size:]])
    X_test, y_test = create_sequences(test_combined, 60)

    with st.spinner("Đang huấn luyện mô hình LSTM..."):
        try:
            tf.keras.utils.set_random_seed(42)
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                LSTM(50),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0,
            )
            test_pred_scaled = model.predict(X_test, verbose=0)
            test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

            rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
            mae = float(mean_absolute_error(y_test_real, test_pred))
            r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

            st.subheader("2. Đánh giá mô hình")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{rmse:,.4f}")
            m2.metric("MAE", f"{mae:,.4f}")
            m3.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "Không xác định")
            st.caption("RMSE và MAE càng thấp thường càng tốt; R² cần được diễn giải cùng bối cảnh dữ liệu.")

            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                st.markdown("**Loss huấn luyện và kiểm định**")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(history.history.get("loss", []), label="Train Loss")
                ax.plot(history.history.get("val_loss", []), label="Validation Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE loss")
                ax.legend()
                ax.grid(alpha=0.25)
                st.pyplot(fig)
                plt.close(fig)

            with hist_col2:
                st.markdown("**Giá thực tế và giá dự báo trên tập kiểm tra**")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(y_test_real, label="Giá thực tế")
                ax.plot(test_pred, label="Giá dự báo")
                ax.set_xlabel("Phiên kiểm tra")
                ax.set_ylabel("Giá")
                ax.legend()
                ax.grid(alpha=0.25)
                st.pyplot(fig)
                plt.close(fig)

            st.subheader(f"3. Dự báo {forecast_days} phiên giao dịch tiếp theo")
            # Recursive multi-step forecast from the last 60 available observations.
            window = all_scaled[-60:].reshape(1, 60, 1)
            future_scaled = []
            for _ in range(forecast_days):
                next_value = model.predict(window, verbose=0)[0, 0]
                future_scaled.append(next_value)
                window = np.concatenate(
                    [window[:, 1:, :], np.array(next_value).reshape(1, 1, 1)],
                    axis=1,
                )

            future_prices = scaler.inverse_transform(
                np.array(future_scaled).reshape(-1, 1)
            ).ravel()

            # Approximate future trading dates using business days; exchange holidays are not excluded.
            last_date = pd.Timestamp(df.index[-1])
            future_dates = pd.bdate_range(
                start=last_date + pd.offsets.BDay(1),
                periods=forecast_days,
            )
            forecast_df = pd.DataFrame({
                "Ngày dự báo (ước tính)": future_dates.strftime("%Y-%m-%d"),
                "Giá dự báo": future_prices,
            })
            forecast_df["Thay đổi so với phiên trước (%)"] = (
                forecast_df["Giá dự báo"].pct_change().fillna(
                    (future_prices[0] / latest_price - 1) if latest_price else np.nan
                ) * 100
            )

            final_change = ((future_prices[-1] / latest_price) - 1) * 100 if latest_price else np.nan
            p1, p2 = st.columns(2)
            p1.metric("Giá dự báo cuối kỳ", f"{future_prices[-1]:,.2f}")
            p2.metric("Thay đổi so với giá gần nhất", f"{final_change:+.2f}%")

            fig, ax = plt.subplots(figsize=(12, 5))
            history_tail = df["Close"].tail(90)
            ax.plot(history_tail.index, history_tail.values, label="Giá lịch sử")
            ax.plot(future_dates, future_prices, marker="o", linestyle="--", label="Dự báo LSTM")
            ax.set_title(f"{ticker}: giá lịch sử và dự báo")
            ax.set_ylabel("Giá")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.autofmt_xdate()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                forecast_df.style.format({
                    "Giá dự báo": "{:,.2f}",
                    "Thay đổi so với phiên trước (%)": "{:+.2f}%",
                }),
                use_container_width=True,
            )

            csv = forecast_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải bảng dự báo CSV",
                data=csv,
                file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
                mime="text/csv",
            )

            st.caption(
                "Ngày dự báo là ngày làm việc ước tính, chưa loại trừ ngày nghỉ lễ của từng sàn. "
                "Dự báo nhiều bước dùng dự báo trước làm đầu vào cho bước tiếp theo nên sai số có thể tích lũy."
            )

        except Exception as e:
            st.error(f"Có lỗi khi huấn luyện hoặc dự báo: {e}")
            st.exception(e)

else:
    st.markdown("### Cách sử dụng")
    st.markdown(
        "1. Nhập mã cổ phiếu theo ký hiệu Yahoo Finance.\n"
        "2. Chọn khoảng thời gian dữ liệu.\n"
        "3. Chọn số phiên muốn dự báo và số epochs huấn luyện.\n"
        "4. Nhấn **Tải dữ liệu và dự báo** để xem biểu đồ, chỉ số đánh giá và tải CSV."
    )
    st.markdown("### Một số mã cổ phiếu Việt Nam để thử")
    st.write("`VNM.VN` · `FPT.VN` · `HPG.VN` · `VCB.VN` · `MWG.VN`")
