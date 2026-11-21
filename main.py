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
st.subheader("나의 혈액 검사 분류 예측 및 결과 근거 분석")
st.write("혈액 검사 수치를 입력하면 예측 분류군과 데이터 기반 설명을 확인할 수 있는 교육용 서비스")

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


# 사용자에게 필요한 예측 기능을 첫 번째 장으로, 수행 과정과 분석 도구를 두 번째 장으로 구분합니다.
tab_user, tab_project = st.tabs([
    "1장. 나의 혈액 검사 분석",
    "2장. 프로젝트 수행 과정"
])

with tab_project:
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
        "일반 헌혈자군과 간 질환 관련 대상자군을 구분하는 분류 모델을 개발하는 것을 "
        "목표로 합니다. 데이터 탐색과 전처리를 바탕으로 이번 단계에서는 "
        "로지스틱 회귀와 랜덤 포레스트 분류 모델을 학습하고 성능을 비교합니다."
    )
    st.info(
        "참고: 'Unnamed: 0'은 원본 데이터의 행 번호이므로 독립 변수에서 제외합니다. "
        "Category를 이진 타깃으로 변환하고 결측치 대체 및 Sex 인코딩을 준비합니다. "
        "모델 학습·평가 단계에서는 학습 데이터에만 전처리 기준을 학습시킵니다."
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
        st.stop()


    # ---------------------------------------------------------
    # 13. 3단계 신규 기능: 머신러닝 모델 학습
    # ---------------------------------------------------------
    st.header("11. 머신러닝 모델 학습")
    st.write(
        "전처리 미리보기용 데이터는 모델 학습에 사용하지 않습니다. "
        "원본 입력 변수 X_raw와 Target y를 사용해 먼저 학습·테스트 데이터를 분리하고, "
        "각 모델의 전처리기는 학습 데이터에만 fit합니다."
    )

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        confusion_matrix
    )
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_validate
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model_ready = False

    try:
        # 2단계에서 만든 원본 독립 변수와 타깃을 사용합니다.
        # processed_features는 전체 데이터에 fit된 미리보기이므로 절대 사용하지 않습니다.
        class_counts_for_split = y.value_counts()
        if len(class_counts_for_split) != 2 or set(class_counts_for_split.index) != {0, 1}:
            raise ValueError("Target에 일반 헌혈자군(0)과 간 질환 관련 대상자군(1)이 모두 있어야 합니다.")
        if int(class_counts_for_split.min()) < 2:
            raise ValueError(
                "소수 클래스 데이터가 2건 미만이어서 계층적 학습·테스트 분할을 수행할 수 없습니다."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        split_summary = pd.DataFrame({
            "구분": ["학습 데이터", "테스트 데이터"],
            "행 개수": [len(X_train), len(X_test)],
            "일반 헌혈자군(0)": [
                int((y_train == 0).sum()), int((y_test == 0).sum())
            ],
            "간 질환 관련 대상자군(1)": [
                int((y_train == 1).sum()), int((y_test == 1).sum())
            ],
            "간 질환 관련 대상자군 비율(%)": [
                round(float((y_train == 1).mean() * 100), 2),
                round(float((y_test == 1).mean() * 100), 2)
            ]
        })

        st.subheader("11-1. 학습 데이터와 테스트 데이터 분리")
        st.dataframe(split_summary, use_container_width=True, hide_index=True)
        st.caption(
            "train_test_split(test_size=0.2, random_state=42, stratify=y)를 적용했습니다. "
            "두 모델은 동일한 분할 데이터를 사용합니다."
        )

        # 로지스틱 회귀: 수치형 중앙값 대체 + 표준화, Sex 최빈값 대체 + 원-핫 인코딩
        def build_logistic_preprocessor():
            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            return ColumnTransformer(
                transformers=[
                    ("numeric", numeric_pipeline, NUMERIC_INPUTS),
                    ("categorical", categorical_pipeline, CATEGORICAL_INPUTS)
                ],
                remainder="drop"
            )

        # 랜덤 포레스트: 트리 모델이므로 수치형 표준화는 하지 않고 결측치 대체 및 인코딩 적용
        def build_forest_preprocessor():
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
                    ("categorical", categorical_pipeline, CATEGORICAL_INPUTS)
                ],
                remainder="drop"
            )

        logistic_pipeline = Pipeline(steps=[
            ("preprocessor", build_logistic_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        forest_pipeline = Pipeline(steps=[
            ("preprocessor", build_forest_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

        # 예측 버튼을 눌러 Streamlit이 다시 실행되어도 동일한 학습 데이터에서는
        # 모델을 매번 다시 학습하지 않도록 Streamlit 리소스 캐시를 사용합니다.
        @st.cache_resource(show_spinner=False)
        def fit_model_once(_pipeline, train_features, train_target, model_key):
            _pipeline.fit(train_features, train_target)
            return _pipeline

        with st.spinner("로지스틱 회귀와 랜덤 포레스트 모델을 확인하고 있습니다..."):
            logistic_pipeline = fit_model_once(
                logistic_pipeline, X_train, y_train, "hepacheck-logistic-v1"
            )
            forest_pipeline = fit_model_once(
                forest_pipeline, X_train, y_train, "hepacheck-random-forest-v1"
            )

        st.success("두 모델의 학습이 완료되었습니다.")
        st.write(
            "**로지스틱 회귀 설정:** max_iter=2000, class_weight='balanced', "
            "수치형 변수 표준화 적용"
        )
        st.write(
            "**랜덤 포레스트 설정:** n_estimators=200, max_depth=5, "
            "class_weight='balanced', 수치형 변수 표준화 미적용"
        )
        st.caption(
            "두 모델 모두 결측치 대체와 Sex 원-핫 인코딩을 Pipeline 내부에서 수행합니다. "
            "따라서 학습 데이터에 대해서만 fit되고 테스트 데이터에는 transform만 적용됩니다."
        )

        # -----------------------------------------------------
        # 14. 3단계 신규 기능: 모델 성능 평가
        # -----------------------------------------------------
        st.header("12. 모델 성능 평가")

        model_pipelines = {
            "로지스틱 회귀": logistic_pipeline,
            "랜덤 포레스트": forest_pipeline
        }
        evaluation_results = {}
        metric_rows = []
        confusion_matrices = {}

        for model_name, model_pipeline in model_pipelines.items():
            predictions = model_pipeline.predict(X_test)
            probabilities = model_pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predictions)

            # 간 질환 관련 대상자군(Target=1)의 지표
            positive_precision, positive_recall, positive_f1, _ = (
                precision_recall_fscore_support(
                    y_test, predictions,
                    labels=[1],
                    average=None,
                    zero_division=0
                )
            )

            # 전체 클래스 기준 macro / weighted 평균
            macro_values = precision_recall_fscore_support(
                y_test, predictions, average="macro", zero_division=0
            )
            weighted_values = precision_recall_fscore_support(
                y_test, predictions, average="weighted", zero_division=0
            )

            auc_value = (
                roc_auc_score(y_test, probabilities)
                if y_test.nunique() == 2 else float("nan")
            )

            result = {
                "예측값": predictions,
                "확률": probabilities,
                "정확도": float(accuracy),
                "정밀도": float(positive_precision[0]),
                "재현율": float(positive_recall[0]),
                "F1-score": float(positive_f1[0]),
                "ROC-AUC": float(auc_value),
                "Macro 정밀도": float(macro_values[0]),
                "Macro 재현율": float(macro_values[1]),
                "Macro F1-score": float(macro_values[2]),
                "Weighted 정밀도": float(weighted_values[0]),
                "Weighted 재현율": float(weighted_values[1]),
                "Weighted F1-score": float(weighted_values[2])
            }
            evaluation_results[model_name] = result
            confusion_matrices[model_name] = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )

            metric_rows.append({
                "모델": model_name,
                "정확도": result["정확도"],
                "정밀도(Target=1)": result["정밀도"],
                "재현율(Target=1)": result["재현율"],
                "F1-score(Target=1)": result["F1-score"],
                "ROC-AUC": result["ROC-AUC"],
                "Macro F1-score": result["Macro F1-score"],
                "Weighted F1-score": result["Weighted F1-score"]
            })

        metrics_df = pd.DataFrame(metric_rows)

        st.subheader("12-1. 간 질환 관련 대상자군(Target=1) 성능")
        st.write(
            "정밀도는 간 질환 관련 대상자라고 예측한 사례 중 실제 Target=1의 비율이고, "
            "재현율은 실제 Target=1 중 모델이 찾아낸 비율입니다. F1-score는 두 지표의 조화평균입니다."
        )
        positive_metrics_df = metrics_df[
            ["모델", "정밀도(Target=1)", "재현율(Target=1)", "F1-score(Target=1)"]
        ].copy()
        st.dataframe(
            positive_metrics_df.style.format({
                "정밀도(Target=1)": "{:.3f}",
                "재현율(Target=1)": "{:.3f}",
                "F1-score(Target=1)": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("12-2. 전체 클래스 평균 성능")
        average_rows = []
        for model_name, result in evaluation_results.items():
            average_rows.append({
                "모델": model_name,
                "Macro 정밀도": result["Macro 정밀도"],
                "Macro 재현율": result["Macro 재현율"],
                "Macro F1-score": result["Macro F1-score"],
                "Weighted 정밀도": result["Weighted 정밀도"],
                "Weighted 재현율": result["Weighted 재현율"],
                "Weighted F1-score": result["Weighted F1-score"]
            })
        st.dataframe(
            pd.DataFrame(average_rows).style.format({
                column: "{:.3f}" for column in [
                    "Macro 정밀도", "Macro 재현율", "Macro F1-score",
                    "Weighted 정밀도", "Weighted 재현율", "Weighted F1-score"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "Macro 평균은 두 클래스를 동일한 비중으로 평가하고, Weighted 평균은 각 클래스의 "
            "표본 수를 반영합니다. ROC-AUC는 Target=1의 predict_proba 확률을 이용해 계산합니다."
        )

        # -----------------------------------------------------
        # 15. 혼동행렬 분석
        # -----------------------------------------------------
        st.header("13. 혼동행렬 분석")
        st.write(
            "혼동행렬은 실제 정답과 모델 예측을 교차해 보여줍니다. "
            "행은 실제 분류, 열은 모델 예측이며 순서는 0(일반 헌혈자군), 1(간 질환 관련 대상자군)입니다."
        )

        for model_name, matrix in confusion_matrices.items():
            st.subheader(f"{model_name} 혼동행렬")
            matrix_display = pd.DataFrame(
                matrix,
                index=["실제 일반 헌혈자군(0)", "실제 간 질환 관련 대상자군(1)"],
                columns=["예측 일반 헌혈자군(0)", "예측 간 질환 관련 대상자군(1)"]
            )
            fig_cm = px.imshow(
                matrix_display,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
                title=f"{model_name} 혼동행렬",
                labels={"x": "모델 예측", "y": "실제 정답", "color": "건수"}
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = matrix.ravel()
            cm_details = pd.DataFrame({
                "구분": ["True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)"],
                "의미": [
                    "일반 헌혈자를 일반 헌혈자로 올바르게 분류",
                    "일반 헌혈자를 간 질환 관련 대상으로 잘못 분류",
                    "간 질환 관련 대상자를 일반 헌혈자로 놓친 경우",
                    "간 질환 관련 대상자를 간 질환 관련 대상으로 올바르게 분류"
                ],
                "건수": [int(tn), int(fp), int(fn), int(tp)]
            })
            st.dataframe(cm_details, use_container_width=True, hide_index=True)

        st.warning(
            "False Negative(FN)는 실제 간 질환 관련 대상자를 일반 헌혈자군으로 분류한 경우입니다. "
            "이 프로젝트에서는 찾아야 할 대상을 놓치는 결과이므로 특히 주의해야 합니다. "
            "다만 이 모델은 교육용 분류 실험이며 실제 진단 결과가 아닙니다."
        )

        # -----------------------------------------------------
        # 16. 모델별 성능 비교
        # -----------------------------------------------------
        st.header("14. 모델별 성능 비교")
        comparison_display = metrics_df[
            ["모델", "정확도", "정밀도(Target=1)", "재현율(Target=1)", "F1-score(Target=1)", "ROC-AUC"]
        ].copy()
        st.dataframe(
            comparison_display.style.format({
                column: "{:.3f}" for column in comparison_display.columns if column != "모델"
            }),
            use_container_width=True,
            hide_index=True
        )

        comparison_long = comparison_display.melt(
            id_vars="모델",
            var_name="평가 지표",
            value_name="점수"
        )
        fig_comparison = px.bar(
            comparison_long,
            x="평가 지표",
            y="점수",
            color="모델",
            barmode="group",
            text="점수",
            title="로지스틱 회귀와 랜덤 포레스트 성능 비교",
            labels={"평가 지표": "평가 지표", "점수": "점수", "모델": "모델"}
        )
        fig_comparison.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_comparison.update_layout(yaxis=dict(range=[0, 1.08]))
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.markdown(
            "- **정확도:** 전체 테스트 데이터 중 맞게 분류한 비율\n"
            "- **정밀도(Target=1):** 간 질환 관련 대상자라고 예측한 사례 중 실제 대상자의 비율\n"
            "- **재현율(Target=1):** 실제 간 질환 관련 대상자 중 모델이 찾아낸 비율\n"
            "- **F1-score(Target=1):** 정밀도와 재현율을 함께 고려한 조화평균\n"
            "- **ROC-AUC:** 임계값을 바꾸어가며 두 클래스를 구분하는 능력을 종합적으로 나타내는 지표"
        )

        # 실제 계산 결과에 근거한 간단한 비교 설명
        recall_values = {
            name: result["재현율"] for name, result in evaluation_results.items()
        }
        f1_values = {
            name: result["F1-score"] for name, result in evaluation_results.items()
        }
        best_recall_model = max(recall_values, key=recall_values.get)
        best_f1_model = max(f1_values, key=f1_values.get)

        st.subheader("성능 결과 해석")
        st.write(
            f"간 질환 관련 대상자군 재현율은 **{best_recall_model}** 모델이 "
            f"{recall_values[best_recall_model]:.3f}으로 두 모델 중 높았습니다. "
            "재현율이 높을수록 실제 Target=1 사례를 덜 놓치는 경향이 있습니다."
        )
        st.write(
            f"Target=1의 F1-score는 **{best_f1_model}** 모델이 "
            f"{f1_values[best_f1_model]:.3f}으로 두 모델 중 높았습니다. "
            "다만 최종 모델 선택은 재현율, 정밀도, F1-score, ROC-AUC와 "
            "혼동행렬을 함께 검토해야 하며 정확도 하나만으로 결정하지 않는 것이 좋습니다."
        )
        st.info(
            "로지스틱 회귀는 비교적 단순하고 계수 기반 해석이 가능한 선형 분류 모델이며, "
            "랜덤 포레스트는 여러 결정 트리로 비선형 관계를 학습할 수 있습니다. "
            "현재 결과는 이 데이터와 한 번의 분할에 대한 실험이므로 다른 데이터에서도 "
            "같은 성능이 나온다고 보장할 수 없습니다."
        )

        # -----------------------------------------------------
        # 17. 학습 데이터 내 5-Fold 교차 검증
        # -----------------------------------------------------
        st.subheader("14-1. 학습 데이터 5-Fold 교차 검증")
        minority_train_count = int(y_train.value_counts().min())
        n_splits = min(5, minority_train_count)

        if n_splits < 2:
            cv_summary = {}
            cv_completed = False
            st.warning(
                "학습 데이터의 소수 클래스 표본이 2건 미만이어서 교차 검증을 수행할 수 없습니다. "
                "독립적인 테스트 데이터는 그대로 최종 평가에 사용했습니다."
            )
            st.header("16. 최종 분류 모델 선정")
            selected_model_name = "로지스틱 회귀"
            selected_pipeline = logistic_pipeline
            st.warning(
                "교차 검증을 수행할 수 없어 최종 우수 모델을 확정하지 못했습니다. "
                "비교 기준 모델인 로지스틱 회귀를 임시 예측 모델로 사용합니다."
            )
        else:
            if n_splits < 5:
                st.warning(
                    f"학습 데이터의 소수 클래스가 {minority_train_count}건이어서 "
                    f"5-Fold 대신 가능한 {n_splits}-Fold 교차 검증을 수행합니다."
                )
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )
            cv_summary = {}
            cv_completed = False

            @st.cache_data(show_spinner=False)
            def cached_cross_validate(_pipeline, train_features, train_target, split_count, model_key):
                local_cv = StratifiedKFold(
                    n_splits=split_count, shuffle=True, random_state=42
                )
                return cross_validate(
                    _pipeline,
                    train_features,
                    train_target,
                    cv=local_cv,
                    scoring={"recall": "recall", "f1": "f1", "roc_auc": "roc_auc"},
                    n_jobs=None,
                    error_score="raise",
                    return_train_score=False
                )

            try:
                cv_rows = []
                scoring_metrics = {
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc"
                }

                for model_name, model_pipeline in model_pipelines.items():
                    cv_result = cached_cross_validate(
                        model_pipeline, X_train, y_train, n_splits, model_name
                    )
                    cv_summary[model_name] = {
                        "재현율 평균": float(pd.Series(cv_result["test_recall"]).mean()),
                        "F1-score 평균": float(pd.Series(cv_result["test_f1"]).mean()),
                        "ROC-AUC 평균": float(pd.Series(cv_result["test_roc_auc"]).mean())
                    }
                    for fold_index in range(n_splits):
                        cv_rows.append({
                            "모델": model_name,
                            "Fold": f"Fold {fold_index + 1}",
                            "재현율": float(cv_result["test_recall"][fold_index]),
                            "F1-score": float(cv_result["test_f1"][fold_index]),
                            "ROC-AUC": float(cv_result["test_roc_auc"][fold_index])
                        })
                    cv_rows.append({
                        "모델": model_name,
                        "Fold": "평균",
                        "재현율": cv_summary[model_name]["재현율 평균"],
                        "F1-score": cv_summary[model_name]["F1-score 평균"],
                        "ROC-AUC": cv_summary[model_name]["ROC-AUC 평균"]
                    })

                cv_df = pd.DataFrame(cv_rows)
                st.dataframe(
                    cv_df.style.format({
                        "재현율": "{:.3f}",
                        "F1-score": "{:.3f}",
                        "ROC-AUC": "{:.3f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                cv_chart_data = cv_df[cv_df["Fold"] != "평균"].melt(
                    id_vars=["모델", "Fold"],
                    var_name="평가 지표",
                    value_name="점수"
                )
                fig_cv = px.bar(
                    cv_chart_data,
                    x="Fold",
                    y="점수",
                    color="모델",
                    barmode="group",
                    facet_row="평가 지표",
                    title=f"학습 데이터 내 {n_splits}-Fold 교차 검증 결과",
                    labels={"Fold": "교차 검증 구간", "점수": "점수"}
                )
                fig_cv.update_yaxes(range=[0, 1.05])
                st.plotly_chart(fig_cv, use_container_width=True)
                st.caption(
                    "교차 검증은 학습 데이터 안에서만 수행했습니다. 테스트 데이터는 "
                    "교차 검증과 모델 선정 기준 계산에 사용하지 않았습니다."
                )
                cv_completed = True
            except Exception as cv_error:
                st.warning(
                    "교차 검증 중 문제가 발생해 교차 검증 기반 모델 선정을 완료하지 못했습니다. "
                    f"원인: {cv_error}"
                )

            # 테스트 데이터 점수로 모델을 고르지 않고, 학습 데이터 교차 검증 결과로 선정합니다.
            # 우선순위: Target=1 재현율 -> Target=1 F1-score -> ROC-AUC
            st.header("16. 최종 분류 모델 선정")
            if cv_completed and len(cv_summary) == 2:
                selected_model_name = max(
                    cv_summary,
                    key=lambda name: (
                        cv_summary[name]["재현율 평균"],
                        cv_summary[name]["F1-score 평균"],
                        cv_summary[name]["ROC-AUC 평균"]
                    )
                )
                selected_pipeline = model_pipelines[selected_model_name]
                selected_cv = cv_summary[selected_model_name]
                st.success(f"교차 검증 기준 최종 선정 모델: {selected_model_name}")
                st.write(
                    "학습 데이터 교차 검증에서 Target=1 재현율을 가장 우선하여 비교하고, "
                    "재현율이 같을 경우 F1-score, 그다음 ROC-AUC를 기준으로 선정했습니다."
                )
                selection_table = pd.DataFrame([
                    {"모델": name, **scores}
                    for name, scores in cv_summary.items()
                ])
                st.dataframe(
                    selection_table.style.format({
                        "재현율 평균": "{:.3f}",
                        "F1-score 평균": "{:.3f}",
                        "ROC-AUC 평균": "{:.3f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "최종 선정은 테스트 데이터의 평가 점수를 기준으로 하지 않았습니다. "
                    "선정된 Pipeline은 학습 데이터 전체로 이미 학습된 객체를 예측에 재사용합니다."
                )
            else:
                # 교차 검증 실패 시 테스트 점수로 임의 선정하지 않습니다.
                # 재현 가능한 기준 모델을 임시 제공하고 화면에 그 한계를 명시합니다.
                selected_model_name = "로지스틱 회귀"
                selected_pipeline = logistic_pipeline
                st.warning(
                    "교차 검증 결과를 얻지 못해 성능 우수 모델을 확정할 수 없습니다. "
                    "비교 기준 모델인 로지스틱 회귀를 임시 예측 모델로 사용합니다. "
                    "이 경우 최종 모델 선정은 잠정적입니다."
                )


        model_ready = True

        # -----------------------------------------------------
        # 18. 랜덤 포레스트 변수 중요도
        # -----------------------------------------------------
        st.header("15. 주요 변수 중요도")
        forest_preprocessor = forest_pipeline.named_steps["preprocessor"]
        forest_classifier = forest_pipeline.named_steps["classifier"]
        encoded_feature_names = forest_preprocessor.get_feature_names_out()
        importance_values = forest_classifier.feature_importances_

        importance_df = pd.DataFrame({
            "변수": encoded_feature_names,
            "중요도": importance_values
        }).sort_values("중요도", ascending=False)

        # 화면에서는 전처리기 내부 접두어를 제거해 변수명을 간결하게 표시합니다.
        importance_df["변수"] = (
            importance_df["변수"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
            .str.replace("Sex_", "Sex = ", regex=False)
        )
        st.dataframe(
            importance_df.assign(중요도=importance_df["중요도"].round(4)),
            use_container_width=True,
            hide_index=True
        )

        fig_importance = px.bar(
            importance_df.sort_values("중요도", ascending=True),
            x="중요도",
            y="변수",
            orientation="h",
            title="랜덤 포레스트 변수 중요도",
            labels={"중요도": "상대적 중요도", "변수": "입력 변수"}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption(
            "변수 중요도는 이 랜덤 포레스트 모델의 분류 과정에서 상대적으로 활용된 정도입니다. "
            "중요도가 높다고 해당 혈액 검사 항목이 질환의 직접적인 원인이라는 뜻은 아닙니다."
        )

        # -----------------------------------------------------
        # 20. 프로젝트 한계 및 의료적 주의사항
        # -----------------------------------------------------
        st.header("18. 프로젝트의 한계 및 의료적 주의사항")
        st.warning(
            "본 서비스는 혈액 검사 데이터를 활용한 교육용 머신러닝 분류 프로젝트입니다. "
            "예측 결과는 학습 데이터의 패턴을 바탕으로 한 분류 결과이며, 실제 의학적 진단이나 "
            "질병 여부를 확정하지 않습니다. 실제 건강 상태는 의료 전문가의 상담과 추가 검사를 "
            "통해 확인해야 합니다.\n\n"
            "또한 Hepatitis, Fibrosis, Cirrhosis를 하나의 Target=1 그룹으로 통합했으며, "
            "의심 헌혈자군(0s)은 이진 분류 학습에서 제외했습니다. 학습 데이터의 규모와 구성에 "
            "따라 예측 성능이 달라질 수 있고, 실제 의료 현장에서 활용하려면 별도의 외부 검증과 "
            "전문가 검토가 필요합니다."
        )

        # -----------------------------------------------------
        # 변수 중요도 및 예측 결과 해석 주의
        # -----------------------------------------------------
        st.caption(
            "모델의 변수 중요도와 예측 결과는 데이터 내 통계적 패턴을 설명하는 참고 정보이며, "
            "인과관계나 개인의 건강 상태를 확정하지 않습니다."
        )

    except Exception as error:
        model_ready = False
        st.error(f"머신러닝 모델 학습 및 평가 중 문제가 발생했습니다: {error}")
        st.info(
            "CSV의 실제 Category 분포와 입력 변수 자료형을 확인해 주세요. "
            "원본 데이터 파일은 수정되지 않았습니다."
        )


with tab_user:
    st.title("나의 혈액 검사 분석")
    st.write(
        "현재 알고 있는 정보의 양에 맞춰 분석 방법을 선택할 수 있습니다. "
        "나이와 성별만으로 시작하고, 알고 있는 혈액 검사 수치가 있다면 선택적으로 추가하세요."
    )

    st.warning(
        "**검사 결과를 활용한 참고 분석 안내**\n\n"
        "입력하신 나이, 성별 및 혈액 검사 수치를 바탕으로 기존 데이터에서 관찰된 분류 패턴을 분석합니다.\n\n"
        "모든 혈액 검사 항목을 입력하지 않아도 일부 정보를 활용한 분석을 이용할 수 있습니다. "
        "다만 입력한 정보가 적을수록 분석에 활용할 수 있는 근거가 제한되며, 결과의 정확도가 낮아질 수 있습니다.\n\n"
        "본 서비스는 교육 및 데이터 분석을 위한 분류 모델이며 의료 진단이나 질병 발생 가능성을 확정하는 도구가 아닙니다. "
        "건강 상태에 대한 정확한 판단과 추가 검사가 필요한 경우 의료진과 상담하시기 바랍니다."
    )

    st.subheader("1. 기본 정보 입력")
    st.caption(
        "나이와 성별은 필수 입력 항목입니다. 혈액 검사 수치는 알고 있는 항목만 선택해 추가할 수 있습니다. "
        "입력하지 않은 항목을 임의의 값으로 채워 예측하지 않습니다."
    )

    help_text = {
        "ALB": "알부민", "ALP": "알칼리성 인산분해효소",
        "ALT": "알라닌 아미노전이효소", "AST": "아스파르테이트 아미노전이효소",
        "BIL": "빌리루빈", "CHE": "콜린에스테라아제",
        "CHOL": "콜레스테롤", "CREA": "크레아티닌",
        "GGT": "감마 글루타밀전이효소", "PROT": "총단백질"
    }
    optional_labs = ["CHOL", "ALT", "AST", "GGT", "ALB", "ALP", "BIL", "CHE", "CREA", "PROT"]

    # 입력 양식: 사용자가 체크한 수치만 실제 모델의 독립 변수에 포함합니다.
    with st.form("adaptive_personal_analysis_form"):
        c1, c2 = st.columns(2)
        with c1:
            age_value = st.number_input("나이(Age)", min_value=1, max_value=120, value=35, step=1)
        with c2:
            observed_sex = sorted(
                modeling_source["Sex"].dropna().astype(str).str.strip().loc[
                    lambda s: s.ne("")
                ].unique().tolist()
            )
            sex_options = observed_sex if observed_sex else ["m", "f"]
            sex_value = st.selectbox(
                "성별(Sex)",
                options=sex_options,
                format_func=lambda x: {"m": "남성", "f": "여성"}.get(x.lower(), x)
            )

        st.markdown("**2. 알고 있는 혈액 검사 수치 선택 입력**")
        st.caption(
            "예: 건강검진 결과지에서 콜레스테롤 수치만 알고 있다면 CHOL만 선택하면 됩니다. "
            "선택한 항목만 별도 분석 모델에 사용됩니다."
        )
        selected_labs = []
        lab_values = {}
        lab_cols = st.columns(2)
        for i, feature in enumerate(optional_labs):
            with lab_cols[i % 2]:
                include_feature = st.checkbox(
                    f"{feature} · {help_text[feature]} 입력",
                    value=False,
                    key=f"include_{feature}"
                )
                lab_value = st.number_input(
                    f"{feature} 수치",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    format="%.3f",
                    key=f"value_{feature}",
                    help=f"검사 결과지의 {help_text[feature]} 값을 입력하세요. 체크한 항목만 모델에 사용됩니다."
                )
                if include_feature:
                    selected_labs.append(feature)
                    lab_values[feature] = lab_value

        submit_analysis = st.form_submit_button(
            "내 분류 결과와 근거 확인",
            type="primary",
            use_container_width=True
        )

    if submit_analysis:
        selected_features = ["Age", "Sex"] + [
            feature for feature in optional_labs if feature in selected_labs
        ]
        user_row = {"Age": float(age_value), "Sex": sex_value}
        user_row.update({feature: float(lab_values[feature]) for feature in selected_labs})
        user_input = pd.DataFrame([user_row], columns=selected_features)

        st.divider()
        st.subheader("3. 분석에 사용한 정보")
        st.write(
            f"입력 정보: 나이, 성별 및 혈액 검사 수치 {len(selected_labs)}개 "
            f"({', '.join(selected_labs) if selected_labs else '추가 검사 수치 없음'})"
        )
        st.dataframe(user_input, use_container_width=True, hide_index=True)

        # 입력 조합과 정확히 같은 변수만으로 모델을 구성합니다.
        # 일부 변수 입력 시 전체 12개 변수 모델에 결측 대체값을 넣어 사용하는 일을 방지합니다.
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import make_scorer, recall_score, f1_score, roc_auc_score
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            subset_source = modeling_source[selected_features + ["Target"]].copy()
            for feature in selected_features:
                if feature != "Sex":
                    subset_source[feature] = pd.to_numeric(
                        subset_source[feature], errors="coerce"
                    )
            subset_source = subset_source.dropna(subset=["Target"])
            class_counts = subset_source["Target"].value_counts()
            min_class_count = int(class_counts.min()) if len(class_counts) == 2 else 0

            if len(class_counts) != 2 or set(class_counts.index) != {0, 1}:
                st.error("선택한 입력 조합으로 두 분류군을 모두 학습할 수 있는 데이터가 없습니다.")
            elif min_class_count < 10:
                st.warning(
                    f"선택한 입력 조합의 최소 분류군 표본 수가 {min_class_count}건입니다. "
                    "안정적인 교차 검증을 위해 각 분류군에 최소 10건이 필요하도록 설정했으며, "
                    "현재는 예측을 제공하지 않습니다."
                )
            else:
                numeric_selected = [f for f in selected_features if f != "Sex"]
                categorical_selected = ["Sex"] if "Sex" in selected_features else []

                numeric_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
                transformers = []
                if numeric_selected:
                    transformers.append(("numeric", numeric_pipeline, numeric_selected))
                if categorical_selected:
                    categorical_pipeline = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
                    transformers.append(("categorical", categorical_pipeline, categorical_selected))

                adaptive_pipeline = Pipeline(steps=[
                    ("preprocessor", ColumnTransformer(
                        transformers=transformers, remainder="drop"
                    )),
                    ("classifier", LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ))
                ])

                X_subset = subset_source[selected_features].copy()
                y_subset = subset_source["Target"].astype(int).copy()
                n_splits = min(5, min_class_count)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                with st.spinner("입력한 항목 조합에 맞는 분류 모델을 검증하고 있습니다..."):
                    cv_result = cross_validate(
                        adaptive_pipeline,
                        X_subset,
                        y_subset,
                        cv=cv,
                        scoring={
                            "recall": "recall",
                            "f1": "f1",
                            "roc_auc": "roc_auc"
                        },
                        error_score="raise"
                    )

                st.success(
                    f"입력한 {len(selected_features)}개 항목 조합의 {n_splits}-Fold 교차 검증을 완료했습니다."
                )
                cv_metrics = pd.DataFrame([{
                    "교차 검증 재현율 평균": float(cv_result["test_recall"].mean()),
                    "교차 검증 F1-score 평균": float(cv_result["test_f1"].mean()),
                    "교차 검증 ROC-AUC 평균": float(cv_result["test_roc_auc"].mean())
                }])
                st.dataframe(
                    cv_metrics.style.format("{:.3f}"),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "위 점수는 현재 CSV 안에서 수행한 교차 검증 결과이며 외부 병원 데이터로 "
                    "검증한 성능이 아닙니다. 입력 조합에 따라 결과가 달라질 수 있습니다."
                )

                mean_cv_auc = float(cv_result["test_roc_auc"].mean())
                if mean_cv_auc <= 0.60:
                    st.warning(
                        f"이 입력 조합의 교차 검증 ROC-AUC 평균은 {mean_cv_auc:.3f}으로, "
                        "분류 성능이 충분하다고 보기 어려워 개인 분류 예측은 제공하지 않습니다. "
                        "대신 입력한 나이·성별과 비슷한 데이터 집단의 분포를 참고 정보로 보여드립니다."
                    )
                    age_mask = (
                        pd.to_numeric(modeling_source["Age"], errors="coerce").between(
                            max(0, float(age_value) - 5), float(age_value) + 5
                        )
                    )
                    sex_mask = (
                        modeling_source["Sex"].astype("string").str.strip()
                        == str(sex_value).strip()
                    )
                    similar_group = modeling_source.loc[age_mask & sex_mask]
                    if len(similar_group) >= 10:
                        similar_counts = similar_group["Target"].value_counts().reindex(
                            [0, 1], fill_value=0
                        )
                        similar_display = pd.DataFrame({
                            "분류군": ["일반 헌혈자군", "간 질환 관련 대상자군"],
                            "데이터 개수": [
                                int(similar_counts.loc[0]), int(similar_counts.loc[1])
                            ],
                            "집단 내 비율(%)": [
                                float(similar_counts.loc[0] / len(similar_group) * 100),
                                float(similar_counts.loc[1] / len(similar_group) * 100)
                            ]
                        })
                        st.write(
                            f"입력한 나이 기준 ±5세, 같은 성별인 데이터 {len(similar_group)}건의 분포입니다."
                        )
                        st.dataframe(
                            similar_display.style.format({"집단 내 비율(%)": "{:.1f}"}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(
                            "입력한 나이(±5세)와 성별에 해당하는 데이터가 10건 미만이어서 "
                            "세부 집단 비율도 표시하지 않습니다."
                        )
                else:
                    adaptive_pipeline.fit(X_subset, y_subset)
                    predicted = int(adaptive_pipeline.predict(user_input)[0])
                    probabilities = adaptive_pipeline.predict_proba(user_input)[0]
                    class_order = adaptive_pipeline.named_steps["classifier"].classes_
                    probability_map = {
                        int(label): float(prob)
                        for label, prob in zip(class_order, probabilities)
                    }
                    p0 = probability_map.get(0, 0.0)
                    p1 = probability_map.get(1, 0.0)

                    st.divider()
                    st.subheader("4. 나의 분류 예측 결과")
                    if predicted == 0:
                        st.success("모델 예측 분류군: 일반 헌혈자군 (Target 0)")
                    else:
                        st.warning("모델 예측 분류군: 간 질환 관련 대상자군 (Target 1)")
                    m1, m2 = st.columns(2)
                    m1.metric("일반 헌혈자군 모델 출력값", f"{p0 * 100:.1f}%")
                    m2.metric("간 질환 관련 대상자군 모델 출력값", f"{p1 * 100:.1f}%")
                    st.caption(
                        "모델 출력값은 학습 데이터에서 관찰된 패턴에 따른 분류 점수이며, "
                        "실제 질병 발생 확률이나 개인의 건강 위험도를 뜻하지 않습니다."
                    )

                    prob_df = pd.DataFrame({
                        "분류군": ["일반 헌혈자군", "간 질환 관련 대상자군"],
                        "모델 출력값(%)": [p0 * 100, p1 * 100]
                    })
                    fig_prob = px.bar(
                        prob_df, x="분류군", y="모델 출력값(%)",
                        text="모델 출력값(%)", range_y=[0, 110],
                        title="두 분류군에 대한 모델 출력값"
                    )
                    fig_prob.update_traces(
                        texttemplate="%{text:.1f}%", textposition="outside"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                    st.divider()
                    st.subheader("5. 입력값과 기존 데이터 분포 비교")
                    st.write(
                        "입력한 수치가 기존 데이터의 두 분류군에서 어떤 위치에 있는지 비교합니다. "
                        "중앙값과 백분위는 통계적 분포 비교일 뿐, 의학적 정상·비정상 판정이 아닙니다."
                    )
                    group0 = modeling_source.loc[modeling_source["Target"] == 0]
                    group1 = modeling_source.loc[modeling_source["Target"] == 1]
                    comparison_rows = []
                    for feature in numeric_selected:
                        vals0 = pd.to_numeric(group0[feature], errors="coerce").dropna()
                        vals1 = pd.to_numeric(group1[feature], errors="coerce").dropna()
                        value = float(user_input.iloc[0][feature])
                        comparison_rows.append({
                            "항목": feature,
                            "입력값": value,
                            "일반 헌혈자군 중앙값": float(vals0.median()) if not vals0.empty else float("nan"),
                            "간 질환 관련 대상자군 중앙값": float(vals1.median()) if not vals1.empty else float("nan"),
                            "일반 헌혈자군 내 백분위(%)": float((vals0 <= value).mean() * 100) if not vals0.empty else float("nan"),
                            "간 질환 관련 대상자군 내 백분위(%)": float((vals1 <= value).mean() * 100) if not vals1.empty else float("nan")
                        })
                    if comparison_rows:
                        comparison_df = pd.DataFrame(comparison_rows)
                        st.dataframe(
                            comparison_df.style.format({
                                c: "{:.2f}" for c in comparison_df.columns if c != "항목"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("수치형 혈액 검사 항목을 추가하지 않아 나이·성별 모델 결과만 표시했습니다.")

                    st.divider()
                    st.subheader("6. 예측 결과에 대한 참고 근거")
                    st.write(
                        "아래 민감도 분석은 입력한 각 수치 또는 성별을 데이터의 대표값으로 하나씩 바꾸었을 때 "
                        "모델 출력이 얼마나 달라지는지 보여줍니다. 인과관계나 질병 원인을 설명하는 것은 아닙니다."
                    )
                    sensitivity_rows = []
                    base_probability = p1
                    for feature in selected_features:
                        altered = user_input.copy()
                        if feature == "Sex":
                            mode_values = subset_source["Sex"].dropna().astype(str).str.strip()
                            if mode_values.empty:
                                continue
                            reference = mode_values.mode().iloc[0]
                            altered.loc[0, feature] = reference
                            reference_text = f"학습 데이터 최빈값 ({reference})"
                        else:
                            reference = float(pd.to_numeric(X_subset[feature], errors="coerce").median())
                            altered.loc[0, feature] = reference
                            reference_text = f"학습 데이터 중앙값 ({reference:.3f})"
                        altered_probability = float(
                            adaptive_pipeline.predict_proba(altered)[0][
                                list(class_order).index(1)
                            ]
                        )
                        sensitivity_rows.append({
                            "항목": feature,
                            "현재 입력값": str(user_input.iloc[0][feature]),
                            "비교 기준값": reference_text,
                            "기준값으로 바꿨을 때 Target 1 출력값 변화(%p)": (
                                altered_probability - base_probability
                            ) * 100
                        })
                    if sensitivity_rows:
                        sensitivity_df = pd.DataFrame(sensitivity_rows)
                        sensitivity_df["절대 변화량"] = sensitivity_df[
                            "기준값으로 바꿨을 때 Target 1 출력값 변화(%p)"
                        ].abs()
                        sensitivity_df = sensitivity_df.sort_values(
                            "절대 변화량", ascending=False
                        ).drop(columns="절대 변화량")
                        st.dataframe(
                            sensitivity_df.style.format({
                                "기준값으로 바꿨을 때 Target 1 출력값 변화(%p)": "{:+.2f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                    outside = []
                    for feature in numeric_selected:
                        observed = pd.to_numeric(X_subset[feature], errors="coerce").dropna()
                        if not observed.empty:
                            low, high = float(observed.min()), float(observed.max())
                            value = float(user_input.iloc[0][feature])
                            if value < low or value > high:
                                outside.append(
                                    f"{feature}: {value:g} (기존 데이터 관측 범위 {low:g}~{high:g})"
                                )
                    if outside:
                        st.warning(
                            "다음 입력값은 기존 데이터의 관측 범위를 벗어납니다. "
                            "모델 결과를 특히 신중하게 해석하세요.\n\n" +
                            "\n".join(f"- {item}" for item in outside)
                        )

        except Exception as prediction_error:
            st.error(f"분류 예측 또는 근거 분석 중 오류가 발생했습니다: {prediction_error}")

    st.divider()
    st.caption(
        "Target 0은 데이터의 일반 헌혈자군, Target 1은 Hepatitis·Fibrosis·Cirrhosis를 통합한 그룹입니다. "
        "의심 헌혈자(0s)는 이진 분류 학습에서 제외했습니다."
    )
