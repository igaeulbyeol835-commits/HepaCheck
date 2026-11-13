import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 2단계 데이터 전처리용 scikit-learn 도구
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

# 기존 요약 화면에서 사용할 한국어 분류명입니다.
# 실제 CSV의 Category 원본 값은 변경하지 않습니다.
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

# 히스토그램과 상자그림에서 선택할 수 있는 수치형 검사 항목입니다.
# 원본 행 번호와 대상자 분류, 성별은 분석 항목에서 제외합니다.
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
        st.info("프로젝트 폴더에 hcvdat0.csv 파일이 있는지 확인해 주세요.")
        st.stop()

    # 원본 CSV를 UTF-8 인코딩으로 읽습니다.
    # 이후 분석에서도 원본 파일에 값을 저장하거나 덮어쓰지 않습니다.
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
    st.error(f"데이터를 불러오는 중 예상하지 못한 오류가 발생했습니다: {error}")
    st.stop()

# 프로젝트에서 사용하는 필수 컬럼이 모두 있는지 확인합니다.
EXPECTED_COLUMNS = [
    "Unnamed: 0", "Category", "Age", "Sex", "ALB", "ALP", "ALT",
    "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]

if missing_columns:
    st.error("CSV 파일에서 필요한 컬럼을 찾지 못했습니다: " + ", ".join(missing_columns))
    st.info("CSV 파일의 첫 행이 컬럼 이름인지, 올바른 데이터 파일인지 확인해 주세요.")
    st.stop()

if len(df) == 0:
    st.error("CSV 파일에 데이터 행이 없습니다. 데이터 내용을 확인해 주세요.")
    st.stop()

if df["Category"].isna().all():
    st.error("Category 컬럼에 분류 정보가 없습니다. 데이터 파일을 확인해 주세요.")
    st.stop()

# 실제 데이터에 등장하는 Category의 순서를 그대로 보관합니다.
# 정렬하거나 임의의 분류를 추가하지 않습니다.
category_order = df["Category"].dropna().drop_duplicates().tolist()

# ---------------------------------------------------------
# 3. 기존 기능: 데이터 기본 정보 숫자 카드
# ---------------------------------------------------------
st.header("1. 데이터 기본 정보")

total_rows = len(df)
total_columns = len(df.columns)
missing_cells = int(df.isna().sum().sum())

# 실제 데이터에 저장된 일반 헌혈자 분류 값을 기준으로 집계합니다.
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
# 4. 기존 기능: Category별 데이터 개수와 비율
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

# 알려진 Category는 한국어로 설명하고, 알 수 없는 값은 원본 값을 표시합니다.
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
# 5. 기존 기능: 컬럼별 자료형 및 결측치 현황 표
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
# 6. 기존 기능: 컬럼 의미 안내
# ---------------------------------------------------------
st.header("4. 컬럼 의미 안내")

description_table = pd.DataFrame({
    "컬럼 이름": list(COLUMN_DESCRIPTIONS.keys()),
    "의미": list(COLUMN_DESCRIPTIONS.values())
})
st.dataframe(description_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 7. 기존 기능: 데이터 미리보기
# ---------------------------------------------------------
st.header("5. 데이터 미리보기")
st.write("아래 표는 원본 데이터의 처음 다섯 행입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 8. 신규 기능 ① 혈액 검사 수치별 히스토그램
# ---------------------------------------------------------
st.header("6. 혈액 검사 수치 분포 분석")
st.write("검사 항목을 선택하면 해당 수치가 어떤 범위에 주로 분포하는지 확인할 수 있습니다.")

# 선택 상자에는 컬럼의 한국어 의미를 함께 표시합니다.
selected_hist_feature = st.selectbox(
    "분포를 확인할 검사 항목을 선택하세요.",
    options=NUMERIC_FEATURES,
    format_func=lambda column: f"{column} - {COLUMN_DESCRIPTIONS[column]}",
    key="histogram_feature"
)

# 원본 데이터는 그대로 두고, 그래프에 필요한 값만 별도로 숫자형으로 변환합니다.
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

    # 표본 수가 적거나 값이 한 종류뿐인 경우에도 그래프가 작동하도록 구간 수를 조절합니다.
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
# 9. 신규 기능 ② Category별 혈액 검사 수치 비교
# ---------------------------------------------------------
st.header("7. Category별 혈액 검사 수치 비교")
st.write("선택한 검사 항목의 분포를 원본 Category별 상자그림으로 비교합니다.")

selected_box_feature = st.selectbox(
    "Category별로 비교할 검사 항목을 선택하세요.",
    options=NUMERIC_FEATURES,
    format_func=lambda column: f"{column} - {COLUMN_DESCRIPTIONS[column]}",
    key="boxplot_feature"
)

# 그래프용 임시 데이터프레임을 만들어 숫자 변환 및 결측치 제외를 적용합니다.
box_data = df[["Category", selected_box_feature]].copy()
box_data[selected_box_feature] = pd.to_numeric(
    box_data[selected_box_feature],
    errors="coerce"
)
box_data = box_data.dropna(subset=["Category", selected_box_feature])

if box_data.empty:
    st.warning("선택한 항목에 Category별로 비교할 수 있는 데이터가 없습니다.")
else:
    # Category 축 순서는 원본 데이터에서 처음 등장한 순서를 유지합니다.
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

    # 그래프에 사용된 표본 수를 Category별로 집계합니다.
    box_counts = (
        box_data.groupby("Category", sort=False)
        .size()
        .reindex(present_category_order, fill_value=0)
        .rename("유효 데이터 개수")
        .reset_index()
    )
    st.write("Category별 표본 개수")
    st.dataframe(box_counts, use_container_width=True, hide_index=True)
    st.caption("상자그림과 표본 개수는 선택한 항목에 결측치가 없는 행을 기준으로 합니다.")

# ---------------------------------------------------------
# 10. 신규 기능 ③ 데이터 결측치 분석
# ---------------------------------------------------------
st.header("8. 데이터 결측치 분석")
st.write("각 컬럼에 결측치가 얼마나 존재하는지 개수와 비율로 확인합니다.")

# 모든 컬럼을 포함해 결측치 현황을 계산합니다.
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

st.write("컬럼별 결측치 현황")
st.dataframe(missing_analysis, use_container_width=True, hide_index=True)

# 결측치가 0개인 컬럼도 빠짐없이 그래프에 포함합니다.
fig_missing = px.bar(
    missing_analysis,
    x="컬럼 이름",
    y="결측치 개수",
    text="결측치 개수",
    labels={
        "컬럼 이름": "컬럼",
        "결측치 개수": "결측치 개수"
    },
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
# 11. 기존 기능: 프로젝트 목적
# ---------------------------------------------------------
st.header("9. 프로젝트 목적")
st.write(
    "이번 프로젝트는 혈액 검사 수치와 대상자의 기본 정보를 이용하여 "
    "일반 헌혈자군과 그 외 대상자군을 구분하는 분류 모델을 개발하는 것을 "
    "목표로 합니다. 이번 단계에서는 모델 개발에 앞서 데이터의 구조, "
    "대상자 분포, 검사 수치별 분포, Category별 차이와 결측치 현황을 탐색합니다."
)
st.info(
    "참고: 'Unnamed: 0'은 원본 데이터의 행 번호이므로 독립 변수에서 제외합니다. "
    "이번 2단계에서는 Category를 이진 타깃으로 변환하고 결측치 대체 및 Sex 인코딩을 "
    "준비합니다. 모델 학습과 예측은 다음 단계에서 진행합니다."
)

# ---------------------------------------------------------
# 12. 2단계 신규 기능: 이진 분류용 데이터 전처리
# ---------------------------------------------------------
st.header("10. 데이터 전처리")
st.write(
    "일반 헌혈자군과 간 질환 관련 대상자군을 구분할 수 있도록 "
    "분류 타깃을 만들고, 결측치 대체와 성별 범주형 인코딩을 준비합니다. "
    "원본 df와 CSV 파일은 변경하지 않습니다."
)

TARGET_MAPPING = {
    "0=Blood Donor": 0,
    "1=Hepatitis": 1,
    "2=Fibrosis": 1,
    "3=Cirrhosis": 1,
    "0": 0, "1": 1, "2": 1, "3": 1,
}
SUSPECT_CATEGORIES = {"0s=suspect Blood Donor", "0s"}
NUMERIC_INPUTS = [
    "Age", "ALB", "ALP", "ALT", "AST", "BIL",
    "CHE", "CHOL", "CREA", "GGT", "PROT"
]
CATEGORICAL_INPUTS = ["Sex"]
FEATURE_COLUMNS = NUMERIC_INPUTS + CATEGORICAL_INPUTS

try:
    preprocessing_required = FEATURE_COLUMNS + ["Category"]
    preprocessing_missing = [col for col in preprocessing_required if col not in df.columns]
    if preprocessing_missing:
        raise ValueError("전처리에 필요한 컬럼이 없습니다: " + ", ".join(preprocessing_missing))

    category_values = df["Category"].astype("string").str.strip()
    if category_values.isna().any():
        raise ValueError(
            "Category에 결측치가 포함되어 있습니다. 정답 그룹을 알 수 없는 행을 "
            "임의 분류하지 않도록 전처리를 중단했습니다. 원본 데이터를 확인해 주세요."
        )

    known_categories = set(TARGET_MAPPING) | SUSPECT_CATEGORIES
    unknown_categories = sorted(set(category_values.astype(str)) - known_categories)
    if unknown_categories:
        raise ValueError(
            "정의되지 않은 Category 값이 발견되었습니다: " + ", ".join(unknown_categories)
        )

    suspect_mask = category_values.isin(SUSPECT_CATEGORIES)
    excluded_suspect_count = int(suspect_mask.sum())
    modeling_source = df.loc[~suspect_mask].copy()
    modeling_source["Target"] = category_values.loc[~suspect_mask].map(TARGET_MAPPING)

    if modeling_source.empty:
        raise ValueError("의심 헌혈자를 제외한 뒤 남은 데이터가 없습니다.")
    if modeling_source["Target"].isna().any():
        raise ValueError("Target을 생성하지 못한 행이 있습니다. Category 값을 확인해 주세요.")

    # 원본 Category, Unnamed: 0, Target은 독립 변수에서 제외합니다.
    X_raw = modeling_source[FEATURE_COLUMNS].copy()
    y = modeling_source["Target"].astype("int64").copy()
    for column in NUMERIC_INPUTS:
        X_raw[column] = pd.to_numeric(X_raw[column], errors="coerce")

    # 실제 Sex 값 확인. 원본 열과 df는 수정하지 않습니다.
    sex_values = modeling_source["Sex"].astype("string").str.strip().replace("", pd.NA)
    X_raw["Sex"] = sex_values.astype(object).where(sex_values.notna(), float("nan"))

    missing_before_count = int(X_raw.isna().sum().sum())
    feature_cell_count = int(X_raw.shape[0] * X_raw.shape[1])
    missing_before_ratio = missing_before_count / feature_cell_count * 100 if feature_cell_count else 0.0

    # 다음 단계에서 train/test 분리 후, 학습 데이터에만 fit할 전처리기 생성 함수
    def build_preprocessor():
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_INPUTS),
                ("categorical", categorical_pipeline, CATEGORICAL_INPUTS),
            ],
            remainder="drop"
        )

    # 화면 확인용 별도 전처리. 전체 데이터 fit 결과이므로 모델 학습에 재사용 금지.
    preview_preprocessor = build_preprocessor()
    transformed_array = preview_preprocessor.fit_transform(X_raw)
    transformed_feature_names = preview_preprocessor.get_feature_names_out()
    if hasattr(transformed_array, "toarray"):
        transformed_array = transformed_array.toarray()

    processed_features = pd.DataFrame(
        transformed_array, columns=transformed_feature_names, index=X_raw.index
    )
    processed_preview = processed_features.copy()
    processed_preview["Target"] = y
    processed_preview["대상자 그룹"] = y.map({0: "일반 헌혈자군", 1: "간 질환 관련 대상자군"})

    missing_after_count = int(processed_features.isna().sum().sum())
    processed_cell_count = int(processed_features.shape[0] * processed_features.shape[1])
    missing_after_ratio = missing_after_count / processed_cell_count * 100 if processed_cell_count else 0.0

    preprocessing_methods = pd.DataFrame([
        {"변수": c, "변수 유형": "수치형", "결측치 처리 방법": "중앙값(median) 대체", "추가 처리": "없음"}
        for c in NUMERIC_INPUTS
    ] + [{
        "변수": "Sex", "변수 유형": "범주형",
        "결측치 처리 방법": "최빈값(most_frequent) 대체",
        "추가 처리": "원-핫 인코딩(One-Hot Encoding)"
    }])

    st.subheader("10-1. 분류 타깃 생성 및 의심 헌혈자 분리")
    c1, c2, c3 = st.columns(3)
    c1.metric("전처리 전 전체 행", f"{len(df):,}건")
    c2.metric("제외된 의심 헌혈자", f"{excluded_suspect_count:,}건")
    c3.metric("이진 분류 대상 행", f"{len(modeling_source):,}건")
    st.info(
        "의심 헌혈자는 일반 헌혈자 또는 간 질환 관련 대상자라고 확정할 수 없는 "
        "별도 분류이므로 Target을 임의 부여하지 않고 이진 분류 데이터에서 제외했습니다. "
        f"제외된 행은 {excluded_suspect_count:,}건이며 원본 데이터에는 그대로 남아 있습니다."
    )
    target_description = pd.DataFrame({
        "원본 Category": ["0=Blood Donor", "1=Hepatitis / 2=Fibrosis / 3=Cirrhosis", "0s=suspect Blood Donor"],
        "새 Target": ["0", "1", "부여하지 않음"],
        "처리": ["일반 헌혈자군", "간 질환 관련 대상자군", "이진 분류 데이터에서 제외"]
    })
    st.dataframe(target_description, use_container_width=True, hide_index=True)

    st.subheader("10-2. Sex 실제 값과 범주 분포")
    sex_distribution = (
        sex_values.fillna("결측").value_counts(dropna=False)
        .rename_axis("Sex 원본 값").reset_index(name="데이터 개수")
    )
    sex_distribution["비율(%)"] = (sex_distribution["데이터 개수"] / len(modeling_source) * 100).round(2)
    st.dataframe(sex_distribution, use_container_width=True, hide_index=True)
    st.caption("Sex 원본 값은 보존하며, 변환 결과에서는 관측된 범주를 기준으로 원-핫 인코딩합니다.")

    st.subheader("10-3. 일반 헌혈자군과 간 질환 관련 대상자군 분포")
    target_counts = y.value_counts().reindex([0, 1], fill_value=0)
    target_distribution = pd.DataFrame({
        "Target": [0, 1],
        "대상자 그룹": ["일반 헌혈자군", "간 질환 관련 대상자군"],
        "데이터 개수": [int(target_counts.loc[0]), int(target_counts.loc[1])]
    })
    target_distribution["비율(%)"] = (target_distribution["데이터 개수"] / len(y) * 100).round(2)
    st.dataframe(target_distribution, use_container_width=True, hide_index=True)
    fig_target = px.bar(
        target_distribution, x="대상자 그룹", y="데이터 개수",
        color="대상자 그룹", text="데이터 개수",
        title="이진 분류 대상자 그룹 분포",
        labels={"대상자 그룹": "대상자 그룹", "데이터 개수": "데이터 개수"}
    )
    fig_target.update_traces(textposition="outside")
    fig_target.update_layout(showlegend=False, yaxis_title="데이터 개수")
    st.plotly_chart(fig_target, use_container_width=True)

    majority_ratio = float(target_distribution["비율(%)"].max())
    if target_counts.min() == 0:
        st.warning("한쪽 대상자 그룹의 데이터가 없습니다. 이진 분류 전에 데이터를 확인해야 합니다.")
    elif majority_ratio >= 80:
        st.warning(
            f"한 집단의 비율이 {majority_ratio:.2f}%로 높아 클래스 불균형 가능성이 큽니다. "
            "다음 모델링 단계에서는 정확도뿐 아니라 정밀도, 재현율, F1-score, 혼동 행렬을 "
            "함께 확인하세요. 이번 단계에서는 데이터를 복제하거나 제거하지 않았습니다."
        )
    else:
        st.info("두 집단의 비율을 확인했습니다. 이번 단계에서는 데이터를 복제하거나 제거하지 않았습니다.")

    st.subheader("10-4. 결측치 대체 전후 비교")
    missing_comparison = pd.DataFrame({
        "구분": ["전처리 전 입력 변수", "전처리 후 변환 변수"],
        "행 개수": [len(X_raw), len(processed_features)],
        "결측 셀 개수": [missing_before_count, missing_after_count],
        "결측 비율(%)": [round(missing_before_ratio, 2), round(missing_after_ratio, 2)]
    })
    st.dataframe(missing_comparison, use_container_width=True, hide_index=True)
    st.caption("전처리 전은 12개 원본 독립 변수 기준, 전처리 후는 대체 및 원-핫 인코딩된 변수 기준입니다.")
    st.write("변수별 전처리 방법")
    st.dataframe(preprocessing_methods, use_container_width=True, hide_index=True)

    st.subheader("10-5. 전처리 결과 요약")
    s1, s2, s3 = st.columns(3)
    s1.metric("전처리 전 행 개수", f"{len(X_raw):,}건")
    s2.metric("전처리 후 행 개수", f"{len(processed_features):,}건")
    s3.metric("전처리 후 결측 셀", f"{missing_after_count:,}개")
    st.write("**독립 변수 후보**")
    st.code(", ".join(FEATURE_COLUMNS), language="text")
    st.write("**종속 변수(Target)**")
    st.write("0 = 일반 헌혈자군, 1 = 간염·섬유화·간경변을 포함한 간 질환 관련 대상자군")
    st.write("**모델 입력에서 제외하는 열**")
    st.write("Category(정답 원본), Unnamed: 0(원본 행 번호), Target(정답 라벨 자체)")
    st.write("**원-핫 인코딩 후 생성된 독립 변수**")
    st.code(", ".join(processed_features.columns.astype(str)), language="text")

    st.subheader("10-6. 최종 전처리 데이터 미리보기")
    st.dataframe(processed_preview.head(10), use_container_width=True, hide_index=True)
    st.warning(
        "데이터 누수 방지 안내: 위 미리보기는 결과 확인을 위해 대상 데이터 전체에 "
        "전처리기를 fit한 별도 예시입니다. 이 미리보기 또는 preview_preprocessor를 "
        "모델 학습·평가에 재사용하면 안 됩니다. 다음 단계에서는 먼저 train_test_split으로 "
        "분리한 뒤 build_preprocessor()로 새 전처리기를 만들고, 학습 데이터에만 "
        "fit_transform, 테스트 데이터에는 transform을 적용해야 합니다."
    )

except Exception as error:
    st.error(f"데이터 전처리 중 문제가 발생했습니다: {error}")
    st.info("원본 CSV 파일은 수정되지 않았습니다. 필수 컬럼, Category 값, Sex 및 검사 수치의 자료형을 확인해 주세요.")
