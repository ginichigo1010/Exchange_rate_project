# 환율 예측 관련 논문 정리

---

## 1. 시계열 모형을 이용한 원/달러 환율 예측모형 비교연구 (서강대학교, 2019)

**출처:** https://journal.scvk.or.kr/media/sites/scvk/2019-007-04/N0890070408/N0890070408.pdf

### 데이터
- 2007년 1월 ~ 2018년 12월, 144개월 월별 원/달러 환율
- 비정상 시계열 특성

### 활용 모형
- **ARIMA(3,1,0)**: Box-Jenkins 방법론으로 모형 식별·추정·진단 후 선정. AIC, SBC 통계량 최소.
- **이중지수 평활법 (Quadratic Exponential Smoothing)**: ARIMA 예측 정확도 비교 검증용

### 결론
- ARIMA(3,1,0)이 잔차 분석상 유의미하며 최적 모형으로 판단
- 두 모형 모두 원/달러 환율 상승 방향으로 예측

---

## 2. Foreign Exchange Forecasting Models: ARIMA and LSTM Comparison (Frontiers, 2025)

**출처:** https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1654093/full

### 데이터
- EUR/USD, GBP/USD, JPY/USD, AUD/USD, NZD/USD + BTC/USD 일별 종가
- 2017-12-18 ~ 2023-01-27

### 방법론
- ADF 테스트로 정상성 확인, 로그 변환 + 차분으로 정상성 확보
- 평가 지표: MAE, MAPE, RMSE

### 모델 비교
| 모델 | 설명 |
|------|------|
| ARIMA | 전통 시계열 모델 |
| LSTM | 장기 의존성 학습, 단기 예측에 강점 |
| 하이브리드 ARIMA-LSTM | ARIMA 예측값을 LSTM 입력으로 활용 |

### 결론
- LSTM이 ARIMA보다 전반적으로 우수. EUR/USD에서 오차 60~70% 감소, BTC/USD·NZD/USD에서 최대 90~97.5% 감소
- 하이브리드 모델은 LSTM 대비 소폭 개선이나 GBP/USD·NZD/USD에서 성능 저하 사례 있음

---

## 3. Foreign Currency Exchange Rate Prediction with Attention-based LSTM (ALFA) (ScienceDirect, 2024)

**출처:** https://www.sciencedirect.com/science/article/pii/S2666827025000313

### 핵심 내용
- **ALFA (Attention-based LSTM)** 모델 제안
- 기존 LSTM에 Attention 메커니즘 추가 → 중요 시점에 가중치 부여
- 적은 수의 피처로도 높은 예측 성능 달성
- 실제 거래 시뮬레이션에서 2,401 USD 수익 창출

---

## 4. Forecasting directional movement of Forex data using LSTM with technical and macroeconomic indicators (Springer, 2020)

**출처:** https://link.springer.com/article/10.1186/s40854-020-00220-2

### 데이터
- **거시경제 지표**: 독일·EU·미국 금리, 인플레이션율, S&P 500, DAX
- **기술적 지표**: MA, MACD, ROC, 모멘텀, RSI, 볼린저 밴드, CCI
- **EUR/USD** 과거 OHLC 데이터

### 모델 구조
- **ME-LSTM**: 거시경제 지표 전용 LSTM
- **TI-LSTM**: 기술적 지표 전용 LSTM
- **하이브리드 모델**: 두 모델 출력을 규칙 기반 의사결정으로 결합

### 독창적 기여
- **3클래스 분류**: 상승 / 하락 / 무변동(no_action) — 무변동 구간 거래 제외로 정확도 향상
- **profit_accuracy 지표**: 전체 중 수익 거래 비율 측정 (무변동 제외)
- 하이브리드 모델이 1일·3일·5일 예측 모두에서 최고 profit_accuracy 달성

---

## 5. Forex forecasting: The critical role of feature selection (ScienceDirect, 2025)

**출처:** https://www.sciencedirect.com/science/article/abs/pii/S1544612325018100

### 실험 설계
- **모델**: 랜덤워크(기준선) / 선형회귀(OLS) / Ridge-RFF(비선형 ML)
- **변수 세트**: 전통 통화 지표 / 확장 지표 / 테일러 준칙 지표
- **훈련 윈도우**: 12개월 / 60개월 / 120개월

### 주요 결과
| 상황 | 우세 모델 |
|------|----------|
| 데이터 적음 (12개월) + 변수 풍부 | Ridge-RFF 소폭 우세 |
| 데이터 보통~많음 (60~120개월) | 선형회귀 우세 |
| 실전 수익성·안정성 | 선형회귀·랜덤워크 더 안정적 |

- Clark-West·Diebold-Mariano 검정 결과, 복잡한 모델의 우위는 통계적으로 대부분 비유의
- **핵심 메시지: 데이터가 충분하면 단순 모델이 경쟁력 있음**

---

## 6. Forex market forecasting using ML: Systematic Literature Review (Journal of Big Data, 2023)

**출처:** https://www.researchgate.net/publication/367511274

### 분석 범위
- 2010~2021년 발표 외환 예측 ML 논문 60편 메타분석

### 주요 발견
- 가장 많이 쓰인 모델: **LSTM, ANN**
- 가장 많이 예측된 통화쌍: **EUR/USD**
- 표준 평가 지표: MAE, RMSE, MAPE, MSE
- **하이브리드 모델 > 단일 모델** (일관된 경향)
- **RNN 계열(LSTM·GRU) > 피드포워드 신경망·SVM** — 금융 시계열에 시간적 패턴이 실재함을 시사

---

## 현재 프로젝트와의 연결

| 논문 인사이트 | 우리 결과와의 관계 |
|--------------|------------------|
| 단순 모델도 충분한 데이터에서 경쟁력 있음 (논문 5) | 선형회귀 R²=0.91로 LSTM 능가한 결과와 일치 |
| 1일 예측에서 LSTM 강점 (논문 2) | LSTM 에포크·튜닝 부족으로 잠재력 미발현 가능성 |
| Attention 메커니즘으로 성능 개선 가능 (논문 3) | ALFA 구조 도입 시 LSTM 성능 향상 여지 |
| 거시+기술 지표 하이브리드 모델 유효 (논문 4) | 현재 9개 피처 중 대부분 기여도 낮음 — 피처 재검토 필요 |
| ARIMA가 월별 데이터에서 유효 (논문 1) | 일별 예측에 ARIMA 추가 비교 고려 가능 |
