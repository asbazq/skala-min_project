# DAY 2 배터리 수명 예측 — MAT부터 학습하는 노트북

사전에 만든 CSV/GZ/분할 목록을 입력으로 요구하지 않는다.
`30-ESSHealth-Day2-Modeling.ipynb` 하나 안에서 아래 전체 과정을 보여준다.

1. MAT 세 파일을 h5py로 읽고 실제 필드 확인
2. 셀별 기록 추출·수명 유효 여부·저자 정제 목록 적용
3. 추출한 내용을 JSON.GZ로 저장
4. 초기 평균·충전 조건·두 파생변수 계산 후 셀별 CSV 저장
5. 프로토콜 그룹 Hold-out과 CV 분할 생성·목록 저장
6. 전처리·모델 후보 비교·최종 학습·Valid/Test 평가

## 실행 준비

원자료 세 파일을 준비하고 노트북 1-1의 RAW_DIR을 그 폴더로 설정한다.

- 2017-05-12_batchdata_updated_struct_errorcorrect.mat
- 2018-02-20_batchdata_updated_struct_errorcorrect.mat
- 2018-04-12_batchdata_updated_struct_errorcorrect.mat

약 8GB의 MAT 원본은 ZIP에 포함하지 않는다. 사용자의 기존 data-30 폴더를 사용한다.
Python 환경에서 `python -m pip install -r DAY2-requirements.txt`를 실행한다.
ZIP에는 같은 목록을 requirements.txt로도 넣었다. h5py가 추가되었으며 검증 버전은 3.16.0이다.
그 외 검증 환경은 Python 3.14.6 / scikit-learn 1.9.0이다.
노트북을 위에서 아래로 실행하거나 `python DAY2-model-learning.py`로 실행한다.

## 학습 순서

13단계, 114셀(코드 57셀)이며 코드 셀은 최대 23줄이다.
1~7단계에서 데이터 생성·X/y·전처리·fit/predict를 먼저 이해한다.
8~13단계에서 후보 선택·중첩 CV·평가·그래프·Self-Practice를 읽는다.
변수 출처는 1-7과 DAY2-변수설명.md에 있다.

## 생성되는 파일

모두 DAY2-learning/ 폴더에 저장한다. 입력으로 미리 준비할 파일이 아니다.

- three_batches_eda_data.json.gz: MAT에서 추출한 곡선·요약 기록
- data_audit.csv: 셀별 제외 이유
- cell_features.csv: 초기 특징과 파생변수 표
- DAY2-split-manifest.csv: 코드에서 생성한 개발·검증·Test 목록
- best-model.joblib, candidate-results.csv, evaluation-predictions.csv, performance-report.csv 등

## 실행 확인

기존 중간 파일이 없는 빈 폴더에서 MAT부터 전체 실행했다.
생성 CSV·분할 목록, 선택 모델, 후보 340개 점수, 평가 예측 86개가 기존 결과와 일치했다.
검증 기록은 DAY2-learning/verification.json에 있다.
Train CV 15.40%, Valid 11.87%, Batch 2 28.08%, Batch 3 18.85%다.
선택용 CV 8.30%와 Test를 구분하고, 과제 기준 9.1% 미달을 그대로 설명한다.

ZIP의 DAY2-learning/은 실제 실행 결과의 예시다. 삭제해도 MAT부터 다시 생성할 수 있다.
루트의 DAY2-results-summary.json은 이전 결과와의 선택적 비교용이며 없으면 비교만 건너뛴다.
보고서는 추가하지 않는다. 현재 결정과 맥락은 PROJECT_LOG.md를 확인한다.

## 평가 기준에 맞춘 설명

4단계에 온도·IR·평균 QD의 보류 근거와 미검증 범위를 명시했다. 9단계는 모델 선택 과정의 중첩 그룹 CV, 10단계는 최종 조합 선택, 11단계는 고정 모델의 Valid/Test 평가다.
DAY 1은 설계 근거와 DAY 2 결과 연결을 갱신했으며 성능 표·Gap 방향을 통일했다. 실제 데이터 처리·학습·예측 계산은 바꾸지 않았다.
