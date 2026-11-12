import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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
    "참고: 'Unnamed: 0'은 원본 데이터의 행 번호이므로 이후 모델의 "
    "독립 변수에서 제외할 예정입니다. 이 화면에서는 원본 구조 확인을 위해 "
    "그대로 표시합니다. 아직 결측치 대체, 이상치 제거, 인코딩, 모델 학습 및 "
    "예측은 수행하지 않습니다."
)
