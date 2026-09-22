import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🌳",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42

# 속성 이름(영어 열 이름) <-> 우리말 이름
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURE_COLS = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

st.title("🌳 분류 모델")
st.caption("stroke(뇌졸중)를 양성(1)으로 두고, 로지스틱 회귀와 의사결정트리로 예측해 봐요.")

st.divider()

# -----------------------------
# 1. 입력 속성 고르기
# -----------------------------
st.subheader("① 입력으로 사용할 속성 고르기")

default_labels = [KOR[c] for c in FEATURE_COLS if c != "bmi"]
selected_labels = st.multiselect(
    "입력으로 사용할 속성을 골라 주세요 (2개 이상)",
    options=[KOR[c] for c in FEATURE_COLS],
    default=default_labels,
)

# 우리말 이름을 다시 영어 열 이름으로, FEATURE_COLS 순서를 그대로 유지
selected_cols = [c for c in FEATURE_COLS if KOR[c] in selected_labels]

if len(selected_cols) < 2:
    st.warning("속성을 2개 이상 골라야 모델을 만들 수 있어요. 위에서 속성을 더 골라 주세요.")
    st.stop()

st.write("고른 속성:", ", ".join(KOR[c] for c in selected_cols))

st.divider()

# -----------------------------
# 2. 데이터 나누기 (번호순 정렬 -> 10명씩 묶어 앞 3명 테스트)
# -----------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)
position_in_group = df_sorted.index % 10
is_test = position_in_group < 3

test_df = df_sorted[is_test].reset_index(drop=True)
train_pool = df_sorted[~is_test].reset_index(drop=True)

# bmi를 고른 경우에만, 훈련용(train_pool)의 중앙값으로 결측치 채우기
if "bmi" in selected_cols:
    bmi_median_train = train_pool["bmi"].median()
    train_pool = train_pool.copy()
    test_df = test_df.copy()
    train_pool["bmi"] = train_pool["bmi"].fillna(bmi_median_train)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median_train)

# -----------------------------
# 3. 나머지 7명(train_pool)만 가지고 크기 맞추기(언더샘플링)
# -----------------------------
train_major = train_pool[train_pool["stroke"] == 0]
train_minor = train_pool[train_pool["stroke"] == 1]

if len(train_minor) <= len(train_major):
    major_sampled = train_major.sample(n=len(train_minor), random_state=RANDOM_STATE)
    minor_sampled = train_minor
else:
    minor_sampled = train_minor.sample(n=len(train_major), random_state=RANDOM_STATE)
    major_sampled = train_major

train_balanced = (
    pd.concat([major_sampled, minor_sampled])
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

st.caption(
    f"테스트 데이터: {len(test_df):,}명 · 크기 맞추기 전 훈련 후보: {len(train_pool):,}명 "
    f"· 크기 맞추기 후 학습 데이터: {len(train_balanced):,}명"
)

X_train = train_balanced[selected_cols]
y_train = train_balanced["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

# -----------------------------
# 4. 모델 만들기
# -----------------------------
log_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
)
tree_model.fit(X_train, y_train)

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)


def get_accuracies(model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc


log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

st.divider()

# -----------------------------
# 5. 정확도 카드
# -----------------------------
st.subheader("② 모델 정확도")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric("로지스틱 회귀(확률로 답하는 모델)", f"{log_test_acc:.1%}")
    st.caption(f"훈련 정확도 {log_train_acc:.1%}　|　테스트 정확도 {log_test_acc:.1%}")

with card2:
    st.metric("의사결정트리(질문으로 답하는 모델)", f"{tree_test_acc:.1%}")
    st.caption(f"훈련 정확도 {tree_train_acc:.1%}　|　테스트 정확도 {tree_test_acc:.1%}")

with card3:
    st.metric("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델", f"{dummy_test_acc:.1%}")
    st.caption(f"훈련 정확도 {dummy_train_acc:.1%}　|　테스트 정확도 {dummy_test_acc:.1%}")

st.divider()

# -----------------------------
# 6. 산점도 + 로지스틱 회귀 결정 경계 + 의사결정트리 영역
# -----------------------------
st.subheader("③ 산점도로 보는 두 모델")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_label = st.selectbox(
        "가로축", options=[KOR[c] for c in selected_cols], index=0, key="x_axis"
    )
with axis_col2:
    default_y_index = 1 if len(selected_cols) > 1 else 0
    y_label = st.selectbox(
        "세로축", options=[KOR[c] for c in selected_cols], index=default_y_index, key="y_axis"
    )

x_col = [c for c in selected_cols if KOR[c] == x_label][0]
y_col = [c for c in selected_cols if KOR[c] == y_label][0]

if x_col == y_col:
    st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    other_cols = [c for c in selected_cols if c not in (x_col, y_col)]
    fixed_values = {c: float(test_df[c].median()) for c in other_cols}

    x_vals = test_df[x_col]
    y_vals = test_df[y_col]
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    x_pad = (x_max - x_min) * 0.05 or 1.0
    y_pad = (y_max - y_min) * 0.05 or 1.0
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    GRID_N = 120

    def linspace(start, end, n):
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    x_grid = linspace(x_min, x_max, GRID_N)
    y_grid = linspace(y_min, y_max, GRID_N)

    # 의사결정트리가 나눈 칸(배경) 계산
    grid_rows = []
    for gy in y_grid:
        for gx in x_grid:
            row = {x_col: gx, y_col: gy}
            row.update(fixed_values)
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)[selected_cols]
    grid_pred = tree_model.predict(grid_df)
    z = [list(grid_pred[i * GRID_N:(i + 1) * GRID_N]) for i in range(GRID_N)]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=z,
            zmin=0,
            zmax=1,
            zsmooth=False,
            showscale=False,
            opacity=0.30,
            colorscale=[[0, "rgba(99,110,250,0.9)"], [1, "rgba(239,85,59,0.9)"]],
            hoverinfo="skip",
            name="의사결정트리 영역",
        )
    )

    color_map = {0: "#636EFA", 1: "#EF553B"}
    name_map = {0: "아님", 1: "뇌졸중"}
    for cls in [0, 1]:
        sub = test_df[test_df["stroke"] == cls]
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=f"실제: {name_map[cls]}",
                marker=dict(color=color_map[cls], size=7, line=dict(width=0.5, color="white")),
            )
        )

    # 로지스틱 회귀 결정 경계(확률 0.5)
    coef_map = dict(zip(selected_cols, log_model.coef_[0]))
    intercept = float(log_model.intercept_[0])
    other_terms = sum(coef_map[c] * fixed_values[c] for c in other_cols) if other_cols else 0.0
    cx = coef_map[x_col]
    cy = coef_map[y_col]

    boundary_note = ""
    if abs(cy) > 1e-12:
        line_x = linspace(x_min, x_max, 50)
        line_y = [-(cx * xv + intercept + other_terms) / cy for xv in line_x]
        visible_y = [yv for yv in line_y if y_min <= yv <= y_max]
        if visible_y:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="로지스틱 회귀 결정 경계(0.5)",
                    line=dict(color="black", width=2, dash="dash"),
                )
            )
            if len(visible_y) < len(line_y):
                boundary_note = "⚠️ 결정 경계선의 일부는 그래프 범위 밖으로 벗어나 있어요."
        else:
            boundary_note = "⚠️ 결정 경계선이 그래프 범위 밖에 있어서 보이지 않아요."
    elif abs(cx) > 1e-12:
        x_boundary = -(intercept + other_terms) / cx
        if x_min <= x_boundary <= x_max:
            fig.add_trace(
                go.Scatter(
                    x=[x_boundary, x_boundary],
                    y=[y_min, y_max],
                    mode="lines",
                    name="로지스틱 회귀 결정 경계(0.5)",
                    line=dict(color="black", width=2, dash="dash"),
                )
            )
        else:
            boundary_note = "⚠️ 결정 경계선이 그래프 범위 밖에 있어서 보이지 않아요."
    else:
        boundary_note = "이 두 속성의 계수가 모두 0이라 결정 경계선을 그릴 수 없어요."

    fig.update_layout(
        title="테스트 데이터 산점도 · 의사결정트리 영역 · 로지스틱 회귀 결정 경계",
        xaxis_title=KOR[x_col],
        yaxis_title=KOR[y_col],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    st.plotly_chart(fig, use_container_width=True)

    if other_cols:
        fixed_text = ", ".join(f"{KOR[c]} = {fixed_values[c]:.2f}" for c in other_cols)
        st.caption(f"그림에 없는 속성은 테스트 데이터의 중앙값으로 고정해서 계산했어요: {fixed_text}")
    else:
        st.caption("고른 속성이 가로축·세로축 두 개뿐이라 고정할 속성이 없어요.")

    if boundary_note:
        st.caption(boundary_note)

st.divider()

# -----------------------------
# 7. 의사결정트리 가지 그림
# -----------------------------
st.subheader("④ 의사결정트리가 던진 질문")

tree_ = tree_model.tree_
classes_list = tree_model.classes_.tolist()
idx1 = classes_list.index(1) if 1 in classes_list else None

leaf_nodes = []
used_features = set()
dot_lines = []


def node_info(node_id):
    n_samples = int(tree_.n_node_samples[node_id])
    value = tree_.value[node_id][0]
    n_stroke = int(value[idx1]) if idx1 is not None else 0
    ratio = (n_stroke / n_samples * 100) if n_samples > 0 else 0.0
    return n_samples, n_stroke, ratio


def build_dot(node_id):
    left = tree_.children_left[node_id]
    right = tree_.children_right[node_id]
    n_samples, n_stroke, ratio = node_info(node_id)
    is_leaf = left == -1 and right == -1

    if is_leaf:
        value = tree_.value[node_id][0]
        pred_class = classes_list[list(value).index(max(value))]
        pred_label = "뇌졸중" if pred_class == 1 else "아님"
        fillcolor = "#FADBD8" if pred_class == 1 else "#D6EAF8"
        label = (
            f"답: {pred_label}\\n"
            f"사람 수: {n_samples}명\\n"
            f"뇌졸중: {n_stroke}명 ({ratio:.1f}%)"
        )
        dot_lines.append(f'{node_id} [label="{label}", fillcolor="{fillcolor}"];')
        leaf_nodes.append((node_id, pred_class))
    else:
        feat_idx = tree_.feature[node_id]
        feat_col = selected_cols[feat_idx]
        used_features.add(feat_col)
        threshold = tree_.threshold[node_id]
        question = f"{KOR[feat_col]} ≤ {threshold:.1f} ?"
        label = (
            f"{question}\\n"
            f"사람 수: {n_samples}명\\n"
            f"뇌졸중: {n_stroke}명 ({ratio:.1f}%)"
        )
        dot_lines.append(f'{node_id} [label="{label}", fillcolor="#EAECEE"];')
        build_dot(left)
        build_dot(right)
        dot_lines.append(f'{node_id} -> {left} [label="예"];')
        dot_lines.append(f'{node_id} -> {right} [label="아니요"];')


build_dot(0)

dot_str = "digraph Tree {\n"
dot_str += 'node [shape=box, style=filled, fontsize=11];\n'
dot_str += "\n".join(dot_lines)
dot_str += "\n}"

st.graphviz_chart(dot_str)

leaf_total = len(leaf_nodes)
leaf_no = sum(1 for _, c in leaf_nodes if c == 0)

st.markdown(f"- 답을 내는 마디는 모두 **{leaf_total}칸**이고, 그중 **{leaf_no}칸**이 '아님'이라고 답해요.")
st.markdown("- 고른 속성 가운데 이 나무가 실제로 물어본 것:")
if used_features:
    for c in selected_cols:
        if c in used_features:
            st.markdown(f"    - {KOR[c]}")
else:
    st.markdown("    - 나무가 아무 속성도 묻지 않고 바로 답을 냈어요.")
