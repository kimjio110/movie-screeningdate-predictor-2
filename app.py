import math
from datetime import timedelta

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. 기본 설정
# =========================

st.set_page_config(
    page_title="영화 마지막 상영일 예측 시뮬레이터",
    page_icon="🎬",
    layout="centered"
)

DATA_PATH = "movie_data_cleaned.csv"


# =========================
# 2. 데이터 불러오기
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # 필요한 컬럼만 사용
    use_cols = [
        "release_date",
        "first_day_audience",
        "first_day_sales",
        "first_day_screens",
        "first_day_showings",
        "screening_days",
    ]

    df = df[use_cols].copy()

    # 날짜 변환
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # 숫자 변환
    number_cols = [
        "first_day_audience",
        "first_day_sales",
        "first_day_screens",
        "first_day_showings",
        "screening_days",
    ]

    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측치 제거
    df = df.dropna()

    # 상영일수가 0 이하인 데이터 제거
    df = df[df["screening_days"] > 0]

    return df


# =========================
# 3. 입력변수 만들기
# =========================

def make_features(df):
    release_date = pd.to_datetime(df["release_date"])

    X = pd.DataFrame({
        "release_month": release_date.dt.month,
        "release_day_of_year": release_date.dt.dayofyear,
        "first_day_audience": df["first_day_audience"],
        "first_day_sales": df["first_day_sales"],
        "first_day_screens": df["first_day_screens"],
        "first_day_showings": df["first_day_showings"],
    })

    return X


# =========================
# 4. 모델 학습
# =========================

@st.cache_resource
def train_model(df):
    X = make_features(df)
    y = df["screening_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3
    )

    model.fit(X_train, y_train)

    # 모델 평가용
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    # 최종 모델은 전체 데이터로 다시 학습
    model.fit(X, y)

    return model, mae, rmse, r2


# =========================
# 5. 화면 구성
# =========================

st.title("🎬 영화 마지막 상영일 예측 시뮬레이터")

st.write(
    """
    영화의 개봉 초기 성과를 입력하면  
    **예상 상영 지속일수**와 **예상 마지막 상영일**을 예측합니다.
    """
)

df = load_data()
model, mae, rmse, r2 = train_model(df)

st.divider()

st.subheader("입력값")

release_date = st.date_input("첫 개봉일")

first_day_audience = st.number_input(
    "첫날 관객수",
    min_value=0,
    step=1000,
    value=50000
)

first_day_sales = st.number_input(
    "첫날 매출액",
    min_value=0,
    step=1000000,
    value=500000000
)

first_day_screens = st.number_input(
    "첫날 스크린 수",
    min_value=0,
    step=10,
    value=500
)

first_day_showings = st.number_input(
    "첫날 상영횟수",
    min_value=0,
    step=10,
    value=2000
)


# =========================
# 6. 예측 실행
# =========================

if st.button("예측하기"):
    input_df = pd.DataFrame([{
        "release_month": release_date.month,
        "release_day_of_year": release_date.timetuple().tm_yday,
        "first_day_audience": first_day_audience,
        "first_day_sales": first_day_sales,
        "first_day_screens": first_day_screens,
        "first_day_showings": first_day_showings,
    }])

    predicted_days = model.predict(input_df)[0]

    # 소수점 제거
    predicted_days = round(predicted_days)

    # 최소 1일 이상으로 보정
    predicted_days = max(1, predicted_days)

    predicted_last_date = release_date + timedelta(days=predicted_days)

    st.divider()
    st.subheader("예측 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="예상 상영 지속일수",
            value=f"{predicted_days}일"
        )

    with col2:
        st.metric(
            label="예상 마지막 상영일",
            value=predicted_last_date.strftime("%Y-%m-%d")
        )


# =========================
# 7. 모델 정보
# =========================

with st.expander("모델 정보 보기"):
    st.write("사용 모델: RANDOM FOREST REGRESSOR")
    st.write(f"MAE: {mae:.2f}일")
    st.write(f"RMSE: {rmse:.2f}일")
    st.write(f"R²: {r2:.3f}")
    st.write(f"학습 데이터 수: {len(df)}개")
