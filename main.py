import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="혈액 검사 데이터 분석실",
    page_icon="🩸",
    layout="wide"
)

st.title("혈액 검사 데이터 분석실")
st.write("혈액 검사 데이터의 기본 구조와 분포를 확인하는 화면입니다.")

# 프로젝트 폴더에 있는 CSV 파일 경로
DATA_PATH = Path(__file__).parent / "hcvdat0.csv"

# 컬럼의 의미를 한국어로 설명하기 위한 사전
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
    "GGT": "감마 글루타밀 전이효소",
    "PROT": "총단백질",
    "Unnamed: 0": "원본 데이터의 행 번호",
}

# Category 코드와 화면에 표시할 한국어 명칭
CATEGORY_LABELS = {
    "0=Blood Donor": "일반 헌혈자",
    "0s=suspect Blood Donor": "의심 헌혈자",
    "1=Hepatitis": "간염",
    "2=Fibrosis": "섬유화",
    "3=Cirrhosis": "간경변",
    # 코드만 저장된 파일에도 대응할 수 있도록 코드 형태도 함께 정의합니다.
    "0": "일반 헌혈자",
    "0s": "의심 헌혈자",
    "1": "간염",
    "2": "섬유화",
    "3": "간경변",
}

# ---------------------------------------------------------
# CSV 파일 읽기
# ---------------------------------------------------------
try:
    if not DATA_PATH.exists():
        st.error(
            "데이터 파일을 찾을 수 없습니다. "
            "main.py와 hcvdat0.csv가 같은 폴더에 있는지 확인해 주세요."
        )
        st.stop()

    # 원본 파일을 읽기만 하며, 데이터 자체는 수정하거나 저장하지 않습니다.
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

# 필수 컬럼이 있는지 확인
expected_columns = [
    "Unnamed: 0", "Category", "Age", "Sex", "ALB", "ALP", "ALT",
    "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]
missing_columns = [column for column in expected_columns if column not in df.columns]

if missing_columns:
    st.error(
        "CSV 파일에서 필요한 컬럼을 찾지 못했습니다: "
        + ", ".join(missing_columns)
    )
    st.info("업로드한 데이터의 첫 행이 컬럼 이름인지, 파일이 올바른지 확인해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 1. 데이터 기본 정보 숫자 카드
# ---------------------------------------------------------
st.header("1. 데이터 기본 정보")

total_rows = len(df)
total_columns = len(df.columns)
missing_cells = int(df.isna().sum().sum())

# Category 값이 문자열 또는 숫자 형태로 저장된 경우를 모두 고려합니다.
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
# 2. Category별 데이터 개수와 비율
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
category_counts["비율(%)"] = (category_counts["데이터 개수"] / total_rows * 100).round(2)

display_category = category_counts[["분류 코드", "분류명", "데이터 개수", "비율(%)"]]
st.dataframe(display_category, use_container_width=True, hide_index=True)

# 실제 데이터에 존재하는 분류만 그래프에 표시합니다.
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
# 3. 컬럼별 자료형 및 결측치 현황
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
st.caption("결측치는 삭제하거나 대체하지 않고, 현재 파일에 저장된 상태 그대로 집계했습니다.")

# ---------------------------------------------------------
# 4. 컬럼 의미 안내
# ---------------------------------------------------------
st.header("4. 컬럼 의미 안내")

description_table = pd.DataFrame({
    "컬럼 이름": list(COLUMN_DESCRIPTIONS.keys()),
    "의미": list(COLUMN_DESCRIPTIONS.values())
})
st.dataframe(description_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 5. 데이터 미리보기
# ---------------------------------------------------------
st.header("5. 데이터 미리보기")
st.write("아래 표는 원본 데이터의 처음 다섯 행입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 6. 프로젝트 목적
# ---------------------------------------------------------
st.header("6. 프로젝트 목적")
st.write(
    "이번 프로젝트는 혈액 검사 수치와 대상자의 기본 정보를 이용하여 "
    "일반 헌혈자군과 그 외 대상자군을 구분하는 분류 모델을 개발하는 것을 "
    "목표로 합니다. 현재 단계에서는 모델 개발에 앞서 데이터의 구조, "
    "대상자 분포, 컬럼별 결측치 현황을 확인합니다."
)
st.info(
    "참고: 'Unnamed: 0'은 원본 데이터의 행 번호이므로 이후 모델의 "
    "독립 변수에서 제외할 예정입니다. 이 화면에서는 원본 구조 확인을 위해 "
    "그대로 표시하며, 아직 전처리나 모델 학습은 수행하지 않습니다."
)
