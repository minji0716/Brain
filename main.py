import streamlit as st
import pandas as pd

# -----------------------------
# 기본 설정 (브라우저 탭 제목 & 아이콘)
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# -----------------------------
# 화면 맨 위 제목
# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("건강 데이터를 살펴보고, 뇌졸중을 예측하는 방법을 함께 공부해 봐요.")

st.divider()

# -----------------------------
# 큰 숫자 카드 4개
# -----------------------------
st.subheader("📊 데이터 한눈에 보기")

total_people = len(df)
total_columns = df.shape[1]
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_count / total_people * 100 if total_people > 0 else 0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,} 명")
card2.metric("열 개수", f"{total_columns} 개")
card3.metric("뇌졸중(stroke=1) 인원", f"{stroke_count:,} 명")
card4.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.divider()

# -----------------------------
# 열 설명 표 (우리말 뜻은 학생이 직접 채우기)
# -----------------------------
st.subheader("📋 열(컬럼) 설명표")
st.markdown("**'우리말 뜻'** 칸은 비어 있어요. 교재를 보고 표 안의 칸을 눌러 직접 채워 보세요!")


def describe_value_kind(series: pd.Series) -> str:
    """각 열의 값 종류를 자동으로 요약해서 보여줍니다."""
    non_null = series.dropna()

    if series.name == "id":
        return "숫자(사람마다 고유한 번호)"

    # 범주형(문자) 또는 값의 종류가 적은 경우: 값 목록을 보여줌
    if series.dtype == object or non_null.nunique() <= 10:
        unique_values = sorted(non_null.unique().tolist(), key=lambda x: str(x))
        return ", ".join(str(v) for v in unique_values)

    # 그 외는 연속적인 숫자형으로 보고 범위를 보여줌
    return f"숫자형(연속값), {non_null.min()} ~ {non_null.max()}"


column_info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": ["" for _ in df.columns],
        "값의 종류": [describe_value_kind(df[col]) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

edited_column_info = st.data_editor(
    column_info,
    use_container_width=True,
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해서 이 열이 어떤 뜻인지 적어 보세요.",
            width="medium",
        ),
    },
    key="column_info_editor",
)

st.divider()

# -----------------------------
# 데이터 처음 다섯 줄
# -----------------------------
st.subheader("🔍 데이터 미리 보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# -----------------------------
# 데이터 출처
# -----------------------------
st.subheader("📚 데이터 출처")
st.text_area(
    label="교재를 보고 데이터 출처를 적어 보세요.",
    placeholder="여기에 데이터 출처를 입력하세요.",
    height=100,
    key="data_source",
)
