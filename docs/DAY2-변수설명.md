# DAY 2 변수 설명 — 출처와 역할

이 문서는 원자료 필드와 코드에서 붙인 이름을 구분하는 학습용 사전이다.
**변수 이름이 많다는 것과 모델 입력이 많다는 것은 다르다.**
DAY 2 후보 입력은 기본 4개와 추가 파생변수 2개이며, 최종 모델 입력은 `dq_log_variance` 하나다.

## 1. MAT 원자료에 있던 필드

| 원자료 필드 | 의미 | 추출·저장 방식 |
|---|---|---|
| `cycle_life` | 실제 전체 수명, 타깃 y | 셀당 숫자 하나 |
| `policy_readable` | 충전 프로토콜의 문자 정보 | MATLAB 문자 코드를 문자열로 바꾸고 `policy`로 저장 |
| `summary` | 사이클별 요약 기록 묶음 | 사이클 번호와 QDischarge/QCharge/IR/온도/충전 시간 목록 |
| `cycles` | 각 사이클의 기록 묶음 | 필요한 사이클의 Qdlin만 선택해 캐시에 저장 |
| `Vdlin` | 보간 용량 곡선의 전압 격자 | 원자료에서 꺼냄. 이번 파일은 3.5→2.0V, 1,000점 |
| `Qdlin` | 각 전압 격자에 대응하는 보간 방전 용량 | 원자료 cycles에서 2·10·100사이클을 추출해 키 `'2'`, `'10'`, `'100'`으로 저장 |
| `summary.QDischarge` / `summary.QCharge` | 사이클별 방전 / 충전 용량 | DAY 1 표에서는 각각 QD / QC로 이름을 줄임 |
| `summary.IR` | 사이클별 내부 저항 기록 | 초기 평균 계산에 사용 |
| `summary.Tavg` / `Tmax` / `Tmin` | 사이클별 평균 / 최고 / 최저 온도 | 요약 기록에 보존 |
| `summary.chargetime` | 사이클별 충전 시간 | 초기 평균 계산에 사용 |
| `summary.cycle` | 사이클 번호 | 초기 2~100 구간 선택에 사용 |

`voltage`는 시간마다 측정한 새 전압을 생성한 변수가 아니다. 원자료의 공통 비교 격자 `Vdlin`을 NumPy 배열로 꺼낸 이름이다.
예를 들어 `voltage[0]`이 3.5V면 `q10[0]`과 `q100[0]`은 같은 3.5V에서의 용량이다.
이 둘을 빼 `delta_q[0]`을 만든다. Q 배열 단위는 Ah, 전압은 V다.

```python
voltage = np.asarray(example['Vdlin'])
q10 = np.asarray(example['Qdlin']['10'])
q100 = np.asarray(example['Qdlin']['100'])
delta_q = q100 - q10
```

## 2. 정리 과정에서 추가한 필드·컬럼

| 이름 | 만든 방법·역할 |
|---|---|
| `cell_id` | 배치와 원래 셀 인덱스로 만든 식별자. 예: b1c0 |
| `batch` | 어느 파일에서 추출했는지 나타내는 1/2/3 구분값 |
| `raw_index` | 원자료에서 셀의 위치를 0부터 센 번호 |
| `model_eligible` | 수명 유효성·후속 실험·저자 정제 목록에 따른 모델 사용 표시 |
| `model_exclusion_reason` | 모델 제외 이유를 기록한 문자열 |
| `C1` / `C2` / `SOC_switch` | 정책 문자열의 첫 C-rate / 두 번째 C-rate / 전환 SOC(%)를 숫자로 분리 |
| `mean_QD` / `mean_IR` / `mean_Tavg` / `mean_Tmax` | 사이클 2~100의 해당 유효 측정값 평균. 이번 DAY 2 입력에서는 보류 |
| `mean_chargetime` | 사이클 2~100의 유효 충전 시간 평균. 기본 후보 입력 |
| `dq_log_variance` | log10(var(Q100−Q10)), ddof=0. 추가 파생변수 1 |
| `dq_band_gap` | ΔQ의 2.8~3.1V 평균 − 2.0~2.5V 평균. 추가 파생변수 2 |
| `protocol_group` | C1/C2/SOC_switch 숫자 조합으로 만든 분할 그룹 |
| `partition` | 개발·Hold-out·Test 역할을 기록한 이름 |
| `cv_holdout_fold` | 개발 셀이 어느 바깥 CV 검증 폴드에 속하는지 기록한 번호 |
| `prediction` | 학습된 모델이 예측한 수명 사이클 수 |
| `ape_pct` | 셀 하나의 절대 오차율: abs(예측−실제)/실제×100 |
| `outside_train_life_range` | 실제 수명이 개발 범위를 벗어나는지 표시. 평가 후 진단용이며 X에 넣지 않음 |

정책 예시 `4.8C(80%)-4C`를 분리하면 C1=4.8, SOC_switch=80, C2=4.0이다.
추가 이름을 붙인 식별자·분할·결과 컬럼은 수명 예측 입력으로 쓰지 않는다.

초기 평균은 DAY 1 코드의 처리 규칙을 적용한 뒤 계산했다. QD는 0.01~1.32 범위 밖 값을 결측 처리하고, 평균에는 양수만 포함하며 충전 시간은 60 이하를 사용했다. 이러한 **값 처리 규칙**과 셀 전체의 `model_eligible` 기준은 구분한다. 이 문서는 기존 계산을 설명하며 새로운 임계값을 추가하지 않는다.

## 3. DAY 2 코드에서 붙인 이름 전체

아래는 측정 데이터의 새 컬럼이 아니라, 데이터·설정·모델·중간 결과를 담는 Python 변수다.

### 파일·설정·데이터 표

| 이름 | 설명 |
|---|---|
| `DATA_DIR` | results 폴더 경로. 입력 MAT는 RAW_DIR에서 읽음. |
| `SAVE_DIR` | 학습용 결과를 저장하는 results 폴더. |
| `SEED` | 난수 시드 42. Random Forest처럼 난수를 쓰는 알고리즘의 재현 설정. |
| `TARGET_MAPE` | 과제에서 비교하는 논문 기준 9.1%. 학습 입력이 아님. |
| `df` | MAT에서 만든 셀별 표. 처음은 식별·수명·사용 여부만 있고 3단계에서 초기 평균·파생변수를 합친 129행 표로 바뀜. |
| `split_info` | 5단계 그룹 분할 코드에서 직접 만든 115개 셀의 개발·검증·Test 목록. |
| `cell_counts` | 배치별 EDA 행 개수와 model_eligible=True 개수를 계산한 표. |
| `data` | df와 split_info를 cell_id로 연결한 모델 대상 115행 표. |
| `TARGET` | 정답 컬럼 이름 cycle_life를 담은 문자열. |
| `BASE_FEATURES` | 충전 조건 C1/C2/SOC_switch와 초기 평균 충전 시간의 컬럼 이름 목록. |
| `FEATURE_SETS` | 비교할 입력 조합 5개의 이름과 컬럼 목록을 연결한 사전. |

### 용량 곡선·파생변수 계산

| 이름 | 설명 |
|---|---|
| `batteries` | MAT에서 추출하고 수명 유효성을 확인한 129개 셀 사전 목록. GZ 파일로 저장할 원본 객체. |
| `eligible_batteries` | batteries 중 model_eligible=True인 115개 셀 목록. |
| `example` | 계산을 한 셀부터 설명하기 위해 선택한 첫 모델 대상 셀의 사전. |
| `voltage / v / reference_voltage` | voltage와 v는 원자료 Vdlin을 배열로 꺼낸 이름. reference_voltage는 모든 셀의 전압 격자가 같은지 비교하는 첫 셀의 격자. |
| `q10 / q100` | 10번째·100번째 사이클의 Qdlin 용량 배열. 각 배열은 1,000점. |
| `delta_q / q_delta` | q100−q10. delta_q는 예시 셀, q_delta는 반복문에서 처리 중인 셀의 전압별 차이. |
| `mid_band / low_band / mid / low` | 2.8~3.1V 또는 2.0~2.5V 위치를 표시하는 불리언 배열. mid/low는 전체 셀 반복문에서 같은 역할. 뒤 그래프에서는 low가 축의 최솟값 숫자로 재사용됨. |
| `log_variance / band_gap` | 예시 셀에서 계산한 로그 분산 / 두 전압 구간 평균 차이 숫자. |
| `feature_rows` | 셀별 파생변수 계산 결과 사전을 모으는 리스트. |
| `calculated_features` | EDA 129개 셀에서 직접 계산한 두 파생변수를 cell_id 인덱스로 정리한 표. |

### 분할·전처리·한 모델 학습

| 이름 | 설명 |
|---|---|
| `train / valid / test / test3` | 개발 29개 / Batch 1 Hold-out 7개 / Batch 2 Test 39개 / Batch 3 추가 Test 40개 표. |
| `cell_overlap / protocol_overlap` | Train과 Valid에 중복되는 셀 ID / 충전 프로토콜 그룹의 집합. 비어 있어야 함. |
| `demo_features` | 기본 학습 예시에서 고른 입력 컬럼 목록: dq_log_variance 하나. |
| `X_train / X_valid / X_test` | 각 자료의 입력 컬럼만 꺼낸 2차원 표. 예시에서는 로그 분산 하나만 포함. |
| `y_train / y_valid / y_test` | 각 자료의 실제 cycle_life를 꺼낸 1차원 Series. |
| `imputer` | Train 중앙값을 학습한 결측 대치 객체. |
| `X_train_filled / X_valid_filled` | Train 중앙값으로 결측을 채운 입력 배열. Valid에서 기준을 다시 계산하지 않음. |
| `scaler` | Train에서 평균·표준편차를 계산한 표준화 객체. |
| `X_train_scaled / X_valid_scaled` | Train 기준으로 표준화한 입력 배열. 새 측정값이 아닌 같은 입력의 변환값. |
| `y_train_log` | np.log(y_train)로 변환한 타깃. 두 파생변수의 log10과 구분. |
| `demo_model` | 7단계 설명용 SVR 모델 객체. 이전 최종 설정으로 기본 fit/predict를 연습. |
| `demo_train_pred` | 예시 모델의 개발 셀 예측을 exp로 역변환한 사이클 수. |
| `demo_train_mape` | 학습했던 개발 셀을 다시 예측한 MAPE. 과제의 Train CV와 다름. |

### Pipeline·후보 탐색 설정

| 이름 | 설명 |
|---|---|
| `selector / selectors` | 입력 컬럼을 고르는 ColumnTransformer 객체 / 입력 조합 5개에 대한 객체 목록. |
| `pipeline` | 입력 선택→결측 대치→표준화→모델을 묶은 객체. CV 학습 부분에서 전처리 기준을 새로 계산. |
| `target_options` | 원래 타깃 또는 log/exp 타깃 변환의 공통 탐색 설정 목록. |
| `param_grid / nonlinear_grids` | 모델·입력·설정 후보 전체 목록 / 그중 SVR·Random Forest 후보 목록. 총 340개. |
| `transform / options` | 후보 설정 반복문의 타깃 종류 raw/log / 그 종류에 해당하는 공통 설정 사전. |
| `columns` | 선택 중인 입력 컬럼 이름 목록. 셀 측정값이 아닌 이름 목록. |
| `settings / feature_name` | 선택 설정에서 꺼낸 모델 하이퍼파라미터 사전 / 해당 입력 조합 이름. |

### 교차검증·최종 선택

| 이름 | 설명 |
|---|---|
| `split_audits` | 각 그룹 CV에서 셀·프로토콜 중복이 0인지 기록하는 리스트. 총 23개 분할 점검. |
| `splitter / folds` | GroupKFold 객체 / 객체가 만든 학습·검증 행 번호 쌍 목록. |
| `fit_idx / held_idx` | CV에서 학습에 쓰는 / 남겨서 검증하는 행 번호 배열. |
| `fit_part / held_part` | 분할 점검 함수에서 꺼낸 학습 / 검증 부분 표. |
| `cells / groups` | 분할 점검 함수의 셀 ID / 프로토콜 중복 집합. 비어 있어야 함. |
| `fold` | 현재 처리하는 폴드 번호, 보통 1~5 또는 1~3. |
| `outer_folds / inner_folds / selection_folds` | 보고용 바깥 5-fold / 각 바깥 학습 부분의 안쪽 3-fold / 개발 전체 최종 선택용 3-fold 분할 목록. |
| `outer_train / outer_valid` | 바깥 CV의 학습 부분 / 검증 부분. outer_valid는 따로 고정한 valid 7개와 다름. |
| `search / final_search` | 바깥 반복 안의 후보 선택 GridSearchCV / 개발 29개 전체에서 최종 후보를 선택하는 GridSearchCV. |
| `predicted` | 현재 모델의 예측 배열. CV와 최종 평가 반복문에서 각각 사용. |
| `oof_predictions` | 각 개발 셀을 그 셀이 학습에 없었던 바깥 모델로 예측한 배열. |
| `choice / chosen` | 바깥 폴드의 선택 모델 설명 / 최종 선택 모델 설명 사전. |
| `fold_mape / fold_rows / fold_results` | 현재 바깥 폴드 MAPE / 폴드 결과 리스트 / 그 리스트를 표로 변환한 것. |
| `train_cv_mape / train_cv_std` | 바깥 5개 MAPE의 단순 평균 / 표본 표준편차. 각각 %, %p 단위. |
| `pooled_oof_mape` | 모든 OOF 예측을 모아서 계산한 보조 MAPE. 폴드별 평균과 다를 수 있음. |
| `selection_mape` | 개발 전체에서 가장 좋은 후보의 선택용 CV MAPE. Train CV·Test 점수와 구분. |
| `best_model` | 최종 설정으로 개발 29개를 학습한 모델 객체. 입력 dq_log_variance, 타깃 로그 변환. |

### 후보 비교·모델 저장

| 이름 | 설명 |
|---|---|
| `cv_results` | final_search.cv_results_에 들어 있는 모든 후보 설정·점수의 사전. |
| `candidate_rows / candidates` | 각 후보 설명·점수 리스트 / MAPE 순으로 정렬한 340행 표. |
| `family_best` | 각 모델 종류 안에서 선택용 CV가 가장 낮은 후보를 한 개씩 꺼낸 표. |
| `svr_log / feature_comparison` | SVR·로그 타깃 후보만 남긴 표 / 그 안에서 입력 조합별 최저 점수 표. |
| `params` | 현재 후보의 설정 사전. 측정 데이터가 아닌 알고리즘 설정. |
| `model_path / model_hash` | 학습용 모델 저장 경로 / 저장 파일 SHA-256 문자열. |
| `lock_record` | 평가 전에 고른 설정·학습 셀·모델 해시를 기록한 사전. |

### 평가·오류 진단

| 이름 | 설명 |
|---|---|
| `evaluation_frames / evaluation_scores` | 평가 이름과 표를 연결한 사전 / 각 평가의 MAPE 사전. |
| `partition / frame` | 현재 처리 중인 평가 이름 / 해당 셀 표. 그래프 반복문에서도 사용. |
| `result / prediction_frames / predictions` | 한 평가의 셀별 예측 표 / 평가별 표 리스트 / 모두 이어 붙인 86행 예측 표. |
| `score` | 현재 평가 자료의 MAPE(%). 기존 기록 비교 단계에서도 사용. |
| `valid_mape / test_mape / test3_mape` | Valid / Batch 2 / Batch 3 전체 평가 MAPE(%). |
| `performance` | Train·Valid·Test·Gap을 과제 양식대로 정리한 9행 표. |
| `test_results` | 전체 예측 표 중 Batch 2 결과만 남긴 표. 큰 오차 진단용. |
| `life_min / life_max` | 개발 셀들의 실제 수명 최솟값 / 최댓값. 사후 진단 범위. |
| `range_diagnosis` | 평가별로 실제 수명이 개발 범위 안/밖인지 구분한 셀 수·평균 오차 표. |
| `reference_path / reference` | 이전 결과 JSON 경로 / 읽은 이전 결과 사전. 학습용 재현 검증에 사용. |

### 그래프·반복문에서 잠시 쓰는 이름

| 이름 | 설명 |
|---|---|
| `fig / axes / ax` | 전체 Figure / 여러 좌표축 배열 / 현재 좌표축 객체. |
| `high / low` | 실제·예측 그래프에서 대각선을 그릴 수명 범위 끝값. low는 앞 단계의 전압 마스크 이름과 재사용되어 역할이 바뀜. |
| `b / battery` | 리스트에서 확인 중인 셀의 사전. b는 짧은 반복문 이름, battery는 전체 곡선 확인에서 사용. |
| `file` | 압축 JSON을 여는 동안 사용하는 파일 객체. |
| `column` | 계산한 파생변수와 기존 CSV를 비교할 현재 컬럼 이름. |
| `key / value / name / row / i / _` | 사전 키·값 / 입력 조합 이름 / 결과 한 행 / 후보 번호 / 이번 위치에서 사용하지 않을 값. 문맥별 임시 이름이며 모델 입력이 아님. |

## 4. 함수와 라이브러리 이름

| 함수 | 역할 |
|---|---|
| `make_estimator()` | 학습 전의 새 Pipeline·타깃 변환기를 만듦 |
| `make_group_folds(frame, n_splits, stage)` | 프로토콜 그룹으로 CV 행 번호를 만들고 중복을 점검. frame은 표, n_splits는 폴드 수, stage는 기록 이름 |
| `make_search(folds)` | 해당 분할에서 후보를 비교할 GridSearchCV를 만듦 |
| `mape_percent(actual, predicted)` | 실제·예측 수명의 MAPE(%) 계산 |
| `describe_choice(params)` | 모델 설정 사전에서 읽기 쉬운 모델·입력·변환·하이퍼파라미터 설명을 꺼냄 |

`np`, `pd`, `plt`는 NumPy·pandas·matplotlib의 짧은 이름이다.
`Path`는 파일 경로, `gzip`은 압축 읽기, `json`은 JSON 읽기·쓰기, `joblib`은 모델 저장,
`hashlib`은 파일 해시 계산, `sklearn`은 사용 버전 확인에 사용한다.
`SVR`, `Ridge`, `Pipeline` 등의 import 이름은 라이브러리의 모델·처리 클래스이며 데이터 변수는 아니다.

## 5. 기억할 구분

1. **원자료:** Vdlin/Qdlin/cycle_life 등 이미 제공된 정보.
2. **이름을 붙여 꺼낸 값:** voltage/q10/q100 등 같은 정보를 계산하기 편하게 담은 것.
3. **계산한 특징:** 초기 평균과 두 ΔQ 요약값. 기본 후보와 추가 파생변수를 구분.
4. **변환한 값:** X_train_scaled/y_train_log 등 같은 입력·타깃의 변환값.
5. **절차·결과:** folds/best_model/predictions/MAPE 등 학습·평가를 수행하며 만든 객체.

최종 모델이 보는 X는 `dq_log_variance` 하나이고, 맞혀야 할 y는 `cycle_life`다.
자세한 계산은 [학습용 노트북](../notebooks/30-ESSHealth-Day2-Modeling.ipynb), 원자료 추출은 노트북 1단계를 참고한다.

## 6. MAT부터 생성하는 코드에 추가된 이름

CSV/GZ를 미리 읽는 방식 대신 원자료부터 직접 만들면서 추가한 변수다.

| 이름 | 설명 |
|---|---|
| `RAW_DIR / MAT_FILES / filename / mat_path` | 원자료 폴더 / 배치별 파일 이름 사전 / 처리 중인 이름 / 전체 파일 경로. |
| `mat / batch_group / summary_group / cycle_group` | 읽기 전용 HDF5 파일 / batch 그룹 / 현재 셀 summary 그룹 / 현재 셀 cycles 그룹. |
| `references / dataset / policy_codes` | Qdlin 위치 참조 표 / 참조가 가리키는 실제 데이터 / MATLAB 정책 문자 코드 배열. |
| `raw_batteries / raw_counts / count` | 추출한 원시 셀 139개 목록 / 배치별 원시 개수 사전 / 현재 배치 셀 수. |
| `batch_number / cell_index / index / cycle / x` | 현재 배치 번호 / 셀의 0부터 시작하는 위치 / 제외 목록에서 순회하는 번호 / 추출할 사이클 번호 / 정책 문자 코드의 현재 값. |
| `exclusion_reasons / reason / life` | 제외 셀 ID와 사유의 사전 / 현재 셀 제외 사유 / 현재 셀 실제 수명. |
| `audit_rows / data_audit / cache_path` | 제외 판단 기록 리스트 / 그 기록을 표로 변환한 결과 / 이 실행에서 저장할 GZ 경로. |
| `q10_array / q100_array` | 모든 EDA 셀 파생변수 반복 계산 중인 셀의 10·100사이클 용량 배열. |
| `summary / early / field / values` | summary를 표로 변환한 객체 / 2~100사이클 부분 / 평균을 계산할 컬럼 이름 / 값 처리 후 평균 대상 Series. |
| `match / c1 / c2 / switch` | 정책 문자열 패턴 검색 결과 / 첫 C-rate / 둘째 C-rate / 전환 SOC(%). |
| `feature_records / feature_path` | 초기 평균·충전 조건·파생변수를 합친 셀별 행 리스트 / 생성하는 CSV 저장 경로. |
| `split_features / batch1` | 모델 사용 가능 115개 셀과 그룹 이름의 표 / 그중 Batch 1의 36개 표. |
| `holdout_splitter / development_idx / valid_idx` | 그룹 Hold-out 객체 / 개발 행 번호 / Hold-out 행 번호. |
| `development / holdout / batch2_test / batch3_test / cv_idx` | 생성 중인 개발·Hold-out·Batch 2·Batch 3 표 / 개발 표의 현재 CV 검증 행 번호. |

원자료 읽기 함수 `read_array`, `read_capacity_curve`, `read_battery`는 MAT 참조를 따라 실제 값을 꺼낸다.
`summarize_initial_cycles`는 초기 측정 평균을, `parse_charging_policy`는 정책의 세 숫자를 계산한다.
이 함수들은 모두 노트북 안에 있으며 외부 스크립트를 호출하지 않는다.
`h5py`는 HDF5 읽기 라이브러리다. [공식 사용 안내](https://docs.h5py.org/en/stable/quick.html)를 참고한다.

## 추가: 12-2 예측 곡선 변수

| 변수 | 뜻 |
|---|---|
| curve_feature | 최종 입력 열 이름, dq_log_variance |
| curve_min / curve_max | 그림에 표시할 모델 대상 셀의 입력 최솟값·최댓값 |
| curve_x | 그 사이를 채운 500개 입력 값 |
| curve_input | predict에 전달할 입력 표 |
| curve_prediction | 각 입력에서 고정 모델이 예측한 사이클 수명 |
| train_x_min / train_x_max | 개발 29개 셀의 입력 범위 |
| curve_in_train | 그림의 입력이 개발 범위 안인지 나타내는 표시, 모델 입력 아님 |

## 디렉터리 정리 후 경로 변수

- PROJECT_ROOT: 저장소 루트. 노트북은 루트 또는 notebooks/에서 실행한다.
- DATA_DIR / SAVE_DIR: results 폴더.
- FIGURE_DIR: results/figures 폴더.
- DAY 1의 BASE: results/eda 폴더. DAY 2 최종 파일과 탐색 결과를 구분한다.
