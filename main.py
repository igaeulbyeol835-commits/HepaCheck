import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 머신러닝에 필요한 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="HepaCheck | 혈액 검사 기반 간 질환 분류 시스템",
    page_icon="🩸",
    layout="wide"
)

st.title("HepaCheck")
st.subheader("혈액 검사 기반 간 질환 분류 시스템")
st.write("혈액 검사 수치를 활용한 데이터 분석 및 머신러닝 예측 서비스")

# main.py와 같은 폴더에 있는 CSV 파일을 불러옵니다.
DATA_PATH = Path(__file__).parent / "hcvdat0.csv"

# 컬럼의 의미를 한국어로 설명하기 위한 사전입니다.
COLUMN_DESCRIPTIONS = {
    "Category": "대상자 분류",
    "Age": "나이",
    "Sex": "성별",
    "ALB": "알부민",
    "ALP": "알칼리성 인산분해효소",
    "ALT": "알라닌 아미노전이효소",
    "AST": "아스파르테이트 아미노전이효소",
    "BIL": "빌리루빈",
    "CHE": "콜린에스테라아제",
    "CHOL": "콜레스테롤",
    "CREA": "크레아티닌",
    "GGT": "감마 글루타밀전이효소",
    "PROT": "총단백질",
    "Unnamed: 0": "원본 데이터의 행 번호",
}

CATEGORY_LABELS = {
    "0=Blood Donor": "일반 헌혈자",
    "0s=suspect Blood Donor": "의심 헌혈자",
    "1=Hepatitis": "간염",
    "2=Fibrosis": "섬유화",
    "3=Cirrhosis": "간경변",
    "0": "일반 헌혈자",
    "0s": "의심 헌혈자",
    "1": "간염",
    "2": "섬유화",
    "3": "간경변",
}

NUMERIC_FEATURES = [
    "Age", "ALB", "ALP", "ALT", "AST",
    "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]

# ---------------------------------------------------------
# 2. CSV 파일 읽기 및 기본 검증
# ---------------------------------------------------------
try:
    if not DATA_PATH.exists():
        st.error(
            "데이터 파일을 찾을 수 없습니다. "
            "main.py와 hcvdat0.csv가 같은 폴더에 있는지 확인해 주세요."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH, encoding="utf-8")

except UnicodeDecodeError:
    st.error("CSV 파일을 UTF-8 인코딩으로 읽지 못했습니다. 파일 인코딩을 확인해 주세요.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("CSV 파일에 읽을 수 있는 데이터가 없습니다.")
    st.stop()
except pd.errors.ParserError as error:
    st.error(f"CSV 파일 형식에 문제가 있습니다: {error}")
    st.stop()
except Exception as error:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()

EXPECTED_COLUMNS = [
    "Unnamed: 0", "Category", "Age", "Sex", "ALB", "ALP", "ALT",
    "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]

if missing_columns:
    st.error("CSV 파일에서 필요한 컬럼을 찾지 못했습니다: " + ", ".join(missing_columns))
    st.stop()

if len(df) == 0:
    st.error("CSV 파일에 데이터 행이 없습니다.")
    st.stop()

if df["Category"].isna().all():
    st.error("Category 컬럼에 분류 정보가 없습니다.")
    st.stop()

category_order = df["Category"].dropna().drop_duplicates().tolist()

# ---------------------------------------------------------
# 3. 데이터 기본 정보
# ---------------------------------------------------------
st.header("1. 데이터 기본 정보")

total_rows = len(df)
total_columns = len(df.columns)
missing_cells = int(df.isna().sum().sum())

blood_donor_count = int(
    df["Category"].astype("string").str.strip().isin(
        ["0=Blood Donor", "0"]
    ).sum()
)

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 데이터 수", f"{total_rows:,}건")
card2.metric("전체 컬럼 수", f"{total_columns:,}개")
card3.metric("전체 결측 셀 개수", f"{missing_cells:,}개")
card4.metric("일반 헌혈자 수", f"{blood_donor_count:,}건")

# ---------------------------------------------------------
# 4. Category별 데이터 개수와 비율
# ---------------------------------------------------------
st.header("2. 대상자 분류별 데이터 분포")

category_counts = (
    df["Category"]
    .astype("string")
    .fillna("결측")
    .value_counts(dropna=False)
    .rename_axis("분류 코드")
    .reset_index(name="데이터 개수")
)

category_counts["분류명"] = category_counts["분류 코드"].map(CATEGORY_LABELS)
category_counts["분류명"] = category_counts["분류명"].fillna(category_counts["분류 코드"])
category_counts["비율(%)"] = (
    category_counts["데이터 개수"] / total_rows * 100
).round(2)

display_category = category_counts[
    ["분류 코드", "분류명", "데이터 개수", "비율(%)"]
]
st.dataframe(display_category, use_container_width=True, hide_index=True)

fig_category = px.bar(
    category_counts,
    x="분류명",
    y="데이터 개수",
    text="데이터 개수",
    labels={"분류명": "대상자 분류", "데이터 개수": "데이터 개수"},
    title="대상자 분류별 데이터 개수"
)
fig_category.update_traces(textposition="outside")
fig_category.update_layout(
    xaxis_title="대상자 분류",
    yaxis_title="데이터 개수",
    showlegend=False
)
st.plotly_chart(fig_category, use_container_width=True)

# ---------------------------------------------------------
# 5. 컬럼별 자료형 및 결측치 현황
# ---------------------------------------------------------
st.header("3. 컬럼별 데이터 및 결측치 현황")

missing_summary = pd.DataFrame({
    "컬럼 이름": df.columns,
    "자료형": [str(dtype) for dtype in df.dtypes],
    "결측치 개수": [int(value) for value in df.isna().sum().values],
})
missing_summary["결측치 비율(%)"] = (
    missing_summary["결측치 개수"] / total_rows * 100
).round(2)

st.dataframe(missing_summary, use_container_width=True, hide_index=True)
st.caption("결측치는 삭제하거나 대체하지 않고, 원본 데이터의 현황만 집계했습니다.")

# ---------------------------------------------------------
# 6. 컬럼 의미 안내
# ---------------------------------------------------------
st.header("4. 컬럼 의미 안내")

description_table = pd.DataFrame({
    "컬럼 이름": list(COLUMN_DESCRIPTIONS.keys()),
    "의미": list(COLUMN_DESCRIPTIONS.values())
})
st.dataframe(description_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 7. 데이터 미리보기
# ---------------------------------------------------------
st.header("5. 데이터 미리보기")
st.write("아래 표는 원본 데이터의 처음 다섯 행입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 8. 혈액 검사 수치별 히스토그램
# ---------------------------------------------------------
st.header("6. 혈액 검사 수치 분포 분석")
st.write("검사 항목을 선택하면 해당 수치의 분포를 확인할 수 있습니다.")

selected_hist_feature = st.selectbox(
    "분포를 확인할 검사 항목을 선택하세요.",
    options=NUMERIC_FEATURES,
    format_func=lambda column: f"{column} - {COLUMN_DESCRIPTIONS[column]}",
    key="histogram_feature"
)

hist_values = pd.to_numeric(
    df[selected_hist_feature].copy(),
    errors="coerce"
).dropna()

if hist_values.empty:
    st.warning("선택한 항목에 그래프로 표시할 수 있는 숫자 데이터가 없습니다.")
else:
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("평균", f"{hist_values.mean():,.2f}")
    stat2.metric("중앙값", f"{hist_values.median():,.2f}")
    stat3.metric("최솟값", f"{hist_values.min():,.2f}")
    stat4.metric("최댓값", f"{hist_values.max():,.2f}")

    bin_count = min(30, max(5, int(hist_values.nunique() ** 0.5 * 2)))
    if hist_values.nunique() == 1:
        bin_count = 1

    hist_data = pd.DataFrame({selected_hist_feature: hist_values})
    fig_hist = px.histogram(
        hist_data,
        x=selected_hist_feature,
        nbins=bin_count,
        labels={selected_hist_feature: f"{selected_hist_feature} 수치"},
        title=f"{selected_hist_feature} 수치 분포"
    )
    fig_hist.update_layout(
        xaxis_title=f"{selected_hist_feature} 수치",
        yaxis_title="데이터 개수",
        bargap=0.08
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"유효한 숫자 데이터 {len(hist_values):,}건을 그래프에 사용했습니다. "
        "결측치 및 숫자로 변환할 수 없는 값은 그래프 계산에서만 제외했습니다."
    )

# ---------------------------------------------------------
# 9. Category별 혈액 검사 수치 비교
# ---------------------------------------------------------
st.header("7. Category별 혈액 검사 수치 비교")

selected_box_feature = st.selectbox(
    "Category별로 비교할 검사 항목을 선택하세요.",
    options=NUMERIC_FEATURES,
    format_func=lambda column: f"{column} - {COLUMN_DESCRIPTIONS[column]}",
    key="boxplot_feature"
)

box_data = df[["Category", selected_box_feature]].copy()
box_data[selected_box_feature] = pd.to_numeric(
    box_data[selected_box_feature],
    errors="coerce"
)
box_data = box_data.dropna(subset=["Category", selected_box_feature])

if box_data.empty:
    st.warning("선택한 항목에 Category별로 비교할 수 있는 데이터가 없습니다.")
else:
    present_category_order = [
        category for category in category_order
        if category in box_data["Category"].unique()
    ]

    fig_box = px.box(
        box_data,
        x="Category",
        y=selected_box_feature,
        category_orders={"Category": present_category_order},
        points="outliers",
        labels={
            "Category": "대상자 분류(Category)",
            selected_box_feature: f"{selected_box_feature} 수치"
        },
        title=f"Category별 {selected_box_feature} 수치 분포 비교"
    )
    fig_box.update_layout(
        xaxis_title="대상자 분류(Category)",
        yaxis_title=f"{selected_box_feature} 수치"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    box_counts = (
        box_data.groupby("Category", sort=False)
        .size()
        .reindex(present_category_order, fill_value=0)
        .rename("유효 데이터 개수")
        .reset_index()
    )
    st.write("Category별 표본 개수")
    st.dataframe(box_counts, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 10. 데이터 결측치 분석
# ---------------------------------------------------------
st.header("8. 데이터 결측치 분석")

missing_counts = df.isna().sum()
missing_analysis = pd.DataFrame({
    "컬럼 이름": df.columns,
    "결측치 개수": missing_counts.values.astype(int),
    "결측치 비율(%)": (missing_counts.values / total_rows * 100).round(2)
})

total_missing_cells = int(missing_counts.sum())
columns_with_missing = int((missing_counts > 0).sum())

missing_card1, missing_card2 = st.columns(2)
missing_card1.metric("전체 결측 셀 개수", f"{total_missing_cells:,}개")
missing_card2.metric("결측치가 존재하는 컬럼 수", f"{columns_with_missing:,}개")

st.dataframe(missing_analysis, use_container_width=True, hide_index=True)

fig_missing = px.bar(
    missing_analysis,
    x="컬럼 이름",
    y="결측치 개수",
    text="결측치 개수",
    labels={"컬럼 이름": "컬럼", "결측치 개수": "결측치 개수"},
    title="컬럼별 결측치 개수"
)
fig_missing.update_traces(textposition="outside")
fig_missing.update_layout(
    xaxis_title="컬럼",
    yaxis_title="결측치 개수",
    yaxis=dict(rangemode="tozero")
)
st.plotly_chart(fig_missing, use_container_width=True)

if total_missing_cells == 0:
    st.success("현재 데이터에는 결측치가 없습니다.")
elif columns_with_missing > 0:
    most_missing_row = missing_analysis.loc[
        missing_analysis["결측치 개수"].idxmax()
    ]
    st.info(
        f"결측치가 가장 많은 컬럼은 {most_missing_row['컬럼 이름']}이며, "
        f"{int(most_missing_row['결측치 개수']):,}개입니다."
    )

# ---------------------------------------------------------
# 11. 프로젝트 목적
# ---------------------------------------------------------
st.header("9. 프로젝트 목적")
st.write(
    "이번 프로젝트는 혈액 검사 수치와 대상자의 기본 정보를 이용하여 "
    "일반 헌혈자군과 그 외 대상자군을 구분하는 분류 모델을 개발하는 것을 "
    "목표로 합니다. 이번 단계에서는 데이터의 구조와 분포를 탐색하고 "
    "분류 모델을 학습하여 성능을 비교합니다."
)
st.info(
    "'Unnamed: 0'은 원본 데이터의 행 번호이고 'Category'는 Target 생성에 "
    "사용하므로 모델 입력 변수에서는 제외합니다. 원본 CSV는 수정하지 않습니다."
)

# =========================================================
# 3단계. 머신러닝 분류 모델 개발
# =========================================================

# ---------------------------------------------------------
# 12. Target 생성 및 클래스 불균형 분석
# ---------------------------------------------------------
st.header("10. Target 생성 및 클래스 불균형 분석")
st.write(
    "Blood Donor는 Target 0으로, 나머지 분류는 Target 1로 지정합니다."
)

# 원본 데이터가 변경되지 않도록 복사본을 만듭니다.
model_data = df.copy()

# 원본 Category의 값에 따라 Target을 생성합니다.
category_to_target = {
    "0=Blood Donor": 0,
    "0s=suspect Blood Donor": 1,
    "1=Hepatitis": 1,
    "2=Fibrosis": 1,
    "3=Cirrhosis": 1,
    "0": 0,
    "0s": 1,
    "1": 1,
    "2": 1,
    "3": 1
}

model_data["Target"] = (
    model_data["Category"].astype("string").str.strip().map(category_to_target)
)

if model_data["Target"].isna().any():
    st.error("Category에 지정되지 않은 값이 있어 Target을 만들 수 없습니다.")
    st.stop()

model_data["Target"] = model_data["Target"].astype(int)

# 모델에 사용할 독립 변수입니다.
# Category와 Unnamed: 0은 포함하지 않습니다.
feature_columns = [
    "Age", "Sex", "ALB", "ALP", "ALT", "AST",
    "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]

X = model_data[feature_columns].copy()
y = model_data["Target"].copy()

# 수치형 검사 항목을 숫자로 변환합니다.
for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

target_counts = y.value_counts().sort_index()
target_table = pd.DataFrame({
    "대상자 그룹": ["일반 헌혈자", "그 외 대상자"],
    "Target": [0, 1],
    "데이터 개수": [
        int(target_counts.get(0, 0)),
        int(target_counts.get(1, 0))
    ]
})
target_table["비율(%)"] = (
    target_table["데이터 개수"] / len(y) * 100
).round(2)

st.dataframe(target_table, use_container_width=True, hide_index=True)

fig_target = px.bar(
    target_table,
    x="대상자 그룹",
    y="데이터 개수",
    text="데이터 개수",
    title="Target별 데이터 분포"
)
fig_target.update_traces(textposition="outside")
st.plotly_chart(fig_target, use_container_width=True)

st.warning(
    "두 그룹의 데이터 수에 차이가 있습니다. 이번 모델 비교에서는 "
    "SMOTE 등의 별도 불균형 보정은 적용하지 않습니다."
)

# ---------------------------------------------------------
# 13. 학습 데이터와 테스트 데이터 분리
# ---------------------------------------------------------
st.header("11. 학습 데이터와 테스트 데이터 분리")
st.write("전체 데이터 중 80%는 학습용, 20%는 테스트용으로 사용합니다.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

split_table = pd.DataFrame({
    "구분": ["학습 데이터", "테스트 데이터"],
    "전체 데이터": [len(X_train), len(X_test)],
    "Target 0": [int((y_train == 0).sum()), int((y_test == 0).sum())],
    "Target 1": [int((y_train == 1).sum()), int((y_test == 1).sum())]
})
st.dataframe(split_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 14. 결측치 처리 및 범주형 변수 인코딩
# ---------------------------------------------------------
st.header("12. 결측치 처리 및 범주형 변수 인코딩")

imputation_method = st.selectbox(
    "수치형 결측치 처리 방법을 선택하세요.",
    ["중앙값", "평균"]
)

if imputation_method == "중앙값":
    numeric_imputer = SimpleImputer(strategy="median")
else:
    numeric_imputer = SimpleImputer(strategy="mean")

st.write(
    "수치형 변수는 선택한 방법으로 결측치를 대체하고, "
    "성별(Sex)은 최빈값으로 대체한 뒤 One-Hot Encoding을 적용합니다."
)

# 학습 데이터에만 전처리 기준을 학습합니다.
numeric_preprocessing = Pipeline(steps=[
    ("imputer", numeric_imputer),
    ("scaler", StandardScaler())
])

category_preprocessing = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessing, NUMERIC_FEATURES),
        ("category", category_preprocessing, ["Sex"])
    ]
)

st.caption(
    "전처리 기준은 학습 데이터에서만 학습합니다. 테스트 데이터에는 "
    "학습 데이터에서 정해진 기준을 적용합니다."
)

# ---------------------------------------------------------
# 15. IQR 방식 이상치 탐색
# ---------------------------------------------------------
st.header("13. IQR 방식 이상치 탐색")
st.write("학습 데이터의 사분위 범위를 이용해 이상치를 탐색합니다.")

outlier_results = []

for column in NUMERIC_FEATURES:
    values = X_train[column].dropna()

    if len(values) > 0:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(
            ((values < lower) | (values > upper)).sum()
        )
    else:
        q1 = q3 = iqr = lower = upper = 0
        outlier_count = 0

    outlier_results.append({
        "검사 항목": column,
        "Q1": round(float(q1), 2),
        "Q3": round(float(q3), 2),
        "IQR": round(float(iqr), 2),
        "이상치 개수": outlier_count
    })

outlier_table = pd.DataFrame(outlier_results)
st.dataframe(outlier_table, use_container_width=True, hide_index=True)
st.caption("이상치는 탐색만 하며 데이터에서 삭제하지 않습니다.")

# ---------------------------------------------------------
# 16. 세 가지 분류 모델 학습
# ---------------------------------------------------------
st.header("14. 분류 모델 학습 및 성능 비교")
st.write(
    "로지스틱 회귀, 의사결정나무, 랜덤 포레스트를 학습하고 "
    "같은 테스트 데이터로 성능을 비교합니다."
)

# 세 모델을 준비합니다.
models = {
    "로지스틱 회귀": LogisticRegression(max_iter=1000),
    "의사결정나무": DecisionTreeClassifier(random_state=42),
    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

trained_models = {}
model_results = []
confusion_matrices = {}

# 각 모델에 같은 전처리 방법을 적용합니다.
for model_name, model in models.items():

    # 모델마다 전처리기를 새로 만들어 독립적으로 학습합니다.
    model_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[
                ("imputer", SimpleImputer(
                    strategy="median" if imputation_method == "중앙값" else "mean"
                )),
                ("scaler", StandardScaler())
            ]), NUMERIC_FEATURES),

            ("category", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), ["Sex"])
        ]
    )

    # 전처리와 모델을 하나로 연결합니다.
    model_pipeline = Pipeline(steps=[
        ("preprocessing", model_preprocessor),
        ("model", model)
    ])

    # 학습 데이터로 학습합니다.
    model_pipeline.fit(X_train, y_train)

    # 테스트 데이터로 예측합니다.
    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    # 평가 지표를 계산합니다.
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    model_results.append({
        "모델": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    })

    trained_models[model_name] = model_pipeline
    confusion_matrices[model_name] = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

# 모델 성능 비교 표
result_table = pd.DataFrame(model_results)

st.subheader("모델별 성능 평가 결과")
st.dataframe(
    result_table.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# 성능 비교 그래프
result_graph = result_table.melt(
    id_vars="모델",
    var_name="평가 지표",
    value_name="점수"
)

fig_result = px.bar(
    result_graph,
    x="평가 지표",
    y="점수",
    color="모델",
    barmode="group",
    range_y=[0, 1],
    title="분류 모델별 성능 비교"
)
st.plotly_chart(fig_result, use_container_width=True)

st.caption(
    "Precision, Recall, F1-score는 Target 1을 기준으로 계산했습니다. "
    "정확도뿐 아니라 Target 1의 재현율과 F1-score도 함께 확인해야 합니다."
)

# ---------------------------------------------------------
# 17. 혼동행렬 확인
# ---------------------------------------------------------
st.subheader("혼동행렬 확인")

selected_model_name = st.selectbox(
    "혼동행렬을 확인할 모델을 선택하세요.",
    list(trained_models.keys())
)

selected_matrix = confusion_matrices[selected_model_name]

matrix_table = pd.DataFrame(
    selected_matrix,
    index=["실제 Target 0", "실제 Target 1"],
    columns=["예측 Target 0", "예측 Target 1"]
)

st.dataframe(matrix_table, use_container_width=True)

st.write(
    "혼동행렬은 실제 정답과 모델의 예측 결과를 비교한 표입니다. "
    "특히 실제 Target 1을 Target 0으로 잘못 분류한 데이터가 얼마나 있는지 확인하세요."
)

# ---------------------------------------------------------
# 18. 예측 서비스
# ---------------------------------------------------------
st.header("15. 예측 서비스")
st.warning(
    "이 서비스는 교육 및 데이터 분석을 위한 분류 실습입니다. "
    "실제 간 질환의 진단이나 의료적 판단에 사용할 수 없습니다."
)

st.write("혈액 검사 수치를 입력하면 선택한 머신러닝 모델의 예측 결과를 보여줍니다.")

# 입력 항목의 기본값은 학습 데이터의 중앙값을 사용합니다.
input_values = {}

with st.form("prediction_form"):
    st.subheader("기본 정보")

    sex_options = sorted(
        X_train["Sex"].dropna().astype(str).unique().tolist()
    )

    if len(sex_options) == 0:
        sex_options = ["m", "f"]

    input_values["Sex"] = st.selectbox("성별(Sex)", sex_options)

    st.subheader("혈액 검사 수치")

    input_columns = st.columns(3)

    for index, column in enumerate(NUMERIC_FEATURES):
        default_value = X_train[column].median()

        if pd.isna(default_value):
            default_value = 0.0

        with input_columns[index % 3]:
            input_values[column] = st.number_input(
                f"{column} ({COLUMN_DESCRIPTIONS[column]})",
                value=float(default_value),
                step=0.1,
                format="%.2f"
            )

    predict_button = st.form_submit_button(
        "예측하기",
        use_container_width=True
    )

if predict_button:

    # 입력값을 데이터프레임으로 만듭니다.
    new_data = pd.DataFrame([input_values], columns=feature_columns)

    # 선택한 모델로 예측합니다.
    prediction = trained_models[selected_model_name].predict(new_data)[0]
    probability = trained_models[selected_model_name].predict_proba(new_data)[0]

    if prediction == 0:
        st.success("예측 결과: Target 0 - 일반 헌혈자 그룹")
    else:
        st.warning("예측 결과: Target 1 - 그 외 대상자 그룹")

    probability_table = pd.DataFrame({
        "대상자 그룹": ["일반 헌혈자(Target 0)", "그 외 대상자(Target 1)"],
        "예측 확률(%)": [probability[0] * 100, probability[1] * 100]
    })

    st.dataframe(
        probability_table.style.format({"예측 확률(%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )

    st.write(f"사용한 모델: **{selected_model_name}**")
