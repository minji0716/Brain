import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

st.title("🔍 데이터 탐색")
st.caption("나이, 평균 혈당, 고혈압, 심장병, 체질량지수, 흡연 상태를 여러 각도에서 살펴봐요.")

st.divider()

# -----------------------------
# 1. 나이 · 평균 혈당 분포 히스토그램
# -----------------------------
st.subheader("① 나이 · 평균 혈당 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig_age_hist.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포")
    fig_glucose_hist.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수")
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# -----------------------------
# 2. 뇌졸중 여부에 따른 나이 · 평균 혈당 상자그림 + 평균값 표
# -----------------------------
st.subheader("② 뇌졸중 여부에 따른 나이 · 평균 혈당 비교")

df_box = df.copy()
df_box["뇌졸중 여부"] = df_box["stroke"].map({0: "아님", 1: "뇌졸중"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df_box, x="뇌졸중 여부", y="age", color="뇌졸중 여부", title="나이 비교"
    )
    fig_age_box.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="나이")
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df_box, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부", title="평균 혈당 비교"
    )
    fig_glucose_box.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="평균 혈당")
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_table = (
    df_box.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
    .reset_index()
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 3. 고혈압 · 심장병 유무에 따른 뇌졸중 비율 막대그래프
# -----------------------------
st.subheader("③ 고혈압 · 심장병 유무에 따른 뇌졸중 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    hyper_rate = (
        df.groupby("hypertension")["stroke"].mean().mul(100).reset_index()
    )
    hyper_rate["고혈압"] = hyper_rate["hypertension"].map({0: "없음", 1: "있음"})
    fig_hyper = px.bar(
        hyper_rate, x="고혈압", y="stroke", title="고혈압 유무에 따른 뇌졸중 비율"
    )
    fig_hyper.update_layout(xaxis_title="고혈압", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    heart_rate = (
        df.groupby("heart_disease")["stroke"].mean().mul(100).reset_index()
    )
    heart_rate["심장병"] = heart_rate["heart_disease"].map({0: "없음", 1: "있음"})
    fig_heart = px.bar(
        heart_rate, x="심장병", y="stroke", title="심장병 유무에 따른 뇌졸중 비율"
    )
    fig_heart.update_layout(xaxis_title="심장병", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 4. bmi 결측치가 있는 사람들의 뇌졸중 비율
# -----------------------------
st.subheader("④ 체질량지수(bmi) 결측치와 뇌졸중 비율")

bmi_missing = df[df["bmi"].isna()]
bmi_missing_count = len(bmi_missing)
bmi_missing_rate = bmi_missing["stroke"].mean() * 100 if bmi_missing_count > 0 else 0.0
overall_count = len(df)
overall_rate = df["stroke"].mean() * 100

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi 결측치 있는 사람", "전체 사람"],
        "인원 수": [bmi_missing_count, overall_count],
        "뇌졸중 비율(%)": [round(bmi_missing_rate, 2), round(overall_rate, 2)],
    }
)
st.dataframe(bmi_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.subheader("⑤ 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"].value_counts().rename_axis("흡연 상태").reset_index(name="사람 수")
)
st.dataframe(smoking_table, use_container_width=True, hide_index=True)
