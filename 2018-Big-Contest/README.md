# 2018-Big-Contest
* Predict whether an user will retain or churn by analyzing his in game activity in Blade and Soul, NCSOFT.
* url : http://www.bigcontest.or.kr/
## Requirements
* R 3.5 (or later)
## Packages
* ```library(dplyr)```
* ```library(lubridate)```
* ```library(tibble)```
* ```library(xgboost)```
* ```library(caret)```
## Methods for Training and Prediction
* ```Gradient Boosting```
* ```XGBoost```
## How to use
### Data
* train : 계정 ID 기준 10만 유저의 활동 데이터
  * train_trade.csv : 유저간 거래 이력
  * train_payment.csv : 유저별 주간 결제 금액
  * train_party.csv : 유저간 파티 구성 관계
  * train_label.csv : 학습 대상 계정들의 고유 아이디 및 이탈 여부 레이블 포함
    * retained : 이탈하지 않은 유저
    * week : 1주 이내 이탈 유저
    * month : 2-4주 이내 이탈 유저
    * 2month : 5-8주 이내 이탈 
  * train_guild.csv
  * train_activity.csv : 유저의 인게임 활동량을 일주일 단위로 집계한 데이터
  
  
* test : 계정 ID 기준 4만 유저의 활동 데이터
  * test_trade.csv
  * test_payment.csv
  * test_party.csv
  * test_guild.csv
  * test_activity.csv
### Evaluation
* 기준
  * 4주 이상 접속하지 않으면 이탈로 간주
  * 데이터 제공 시점 시작으로 12주간 접속 이력으로 이탈 여부 결정
* 평가방식
  * 예측 성능 지표: F1 Score
    * 예측 모델이 실제 이탈 고객을 정확하게 맞췄는지, 실제 이탈 고객을 에측 모델이 예측했는지의 조화 평균값
    ![f1score](https://user-images.githubusercontent.com/42960718/52997067-ce0e6500-3462-11e9-866a-1741c55ab575.PNG)
### EDA
* train 데이터의 이탈 여부 레이블이 고르게 분포되어 있음
  * 10만행의 train 데이터가 각 2month 25,000행, month 25,000행, retained 25,000행, week 25,000행으로 분포
* train (train 데이터 7만행) → validation (train 데이터 3만행) → test (test 데이터 4만행) 으로 나누어 학습
### Data Pre-processing
* 데이터가 각각 다른 구조를 지니고 있어 key variable (주요 변수) 인 유저 ID (acc_id) 를 기준으로 merge
### Feature Engineering
* 기존 8주간 데이터별로 요약 변수 생성

#### 참고한 csv 데이터별 파생변수
* train_activity
  * 8주간 총 게임 접속 횟수
  * 8주간 평균 접속 횟수
* train_payment
  * 8주간 평균 결제 금액
* train_party
  * 8주간 총 파티 횟수
  * 8주간 총 파티 시간
* train_guild
  * 8주간 문파 변경 횟수
* train_trade
  * 판매자 계정 아이디 기준
    * 총 아이템 판매 개수
    * 판매 물품 종류 및 종류별 판매 개수
  * 구매자 게정 아이디 기준
    * 총 아이템 구입 개수
    * 구매 물품 종류 및 종류별 구매 개수
  * 거래 활동 시간 기준
    * 아침, 점심, 저녁, 새벽으로 데이터 구간화
    * 주중 (월-목) 및 주말 (금-일) 로 데이터 구간화
    * 시간에 따른 구매 물품 종류 및 종류별 구매 개수
    
    
* total_activity
  * 주별 총 게임 접속 횟수
  * 주별 평균 접속 횟수
* total_payment
  * 주별 평균 결제 금액
  * 줍졀 중 최고 결제 금액
* total_party
  * 주별 총 파티 횟수
  * 주별 총 파티 시간
* total_guild
  * 주별 문파 변경 횟수
* total_trade
  * 판매자 계정 아이디 기준
    * 총 아이템 판매 개수
    * 판매 물품 종류 및 종류별 판매 개수
  * 구매자 계정 아이디 기준
    * 총 아이템 구입 개수
    * 구매 물품 종류 및 종류별 구매 개수
  * 거래 활동 시간 기준
    * 아침, 점심, 저녁, 새벽으로 데이터 구간화
    * 주중 (월-목) 및 주말 (금-일) 로 데이터 구간화
    * 시간에 따른 구매 물품 종류 및 종류별 구매 개수
### Reducing Data Dimension
#### Flatten
 * 유저의 주별 데이터로 인해 불필요하게 늘어난 행을 Flatten 하여 차원을 축소
 
 
![flatten](https://user-images.githubusercontent.com/42960718/52998094-b7b5d880-3465-11e9-8cf2-76412d3c182c.PNG)
#### PCA
* 데이터의 분산을 최대한 보존하는 축을 찾아 고차원의 기존 변수를 조합하고, 저차원의 새로운 변수를 만드는 차원 축소 기법
  * 총 2,117개의 설명 변수를 조합하여 cumulative proportion (변수 설명 누적기여율) 이 95% 가 되는 주성분 k개 탐색
  * 총 2,117개의 설명 변수를 PCA를 활용하여 568개로 축소시켜 데이터 희소성과 연산 속도 문제 개선
### Model
* `XGBoost`
 * 기존 Gradient Boosting 의 연산 속도 문제를 해결하기 위해 연산 속도와 모델 성능에 초점을 둔 패키지
 * C 언어로 작성되어 수행 속도가 빠르고 cross validation 을 자동으로 수행하는 함수가 존재하여 추가적인 교차 검증 코드 불필요
 * 기존에 사용했던 Gradient Boosting 모델보다 연산 속도 및 모델 성능이 더 높게 나옴
### Performance
* ```Accuracy : .7301``` as of 2018-09-07 상위 100위 리더보드 진출


![accuracy](https://user-images.githubusercontent.com/42960718/52998438-a4573d00-3466-11e9-95d4-9c062418f360.png)
### Improvement
* week 와 retained 는 모델이 비교적 정확하게 구분하여 에측 성능이 높음
* 그에 반면 month 와 2month 를 모델이  정확하게 구분하지 못하여 성능이 낮았음
* confusion matrix 를 확인했을 때 각각 month 의 recall 과 precision 이 낮게 나온것으로 보아, month 로 분류해야하는 유저를 2month 로 모델이 오분류 
* 한달 후에 이탈하는 유저와 두달 후에 이탈하는 유저의 활동 양상이 비슷해서 생긴 결과로 유추됨
* 단, 엔씨소프트사의 패치 혹은 컨텐츠 업데이트 주기에 영향을 끼치는 분석 결과인만큼 더욱 정교하게 분류할 수 있는 모델 필요
* 또한, ```XG Boost``` 단일 모델만 사용할 것이 아니라 ```Random Forest``` 같은 decision tree 계열의 모델을 일차적으로 사용하여 학습시킨 후에 XGBoost 를 활용한 stacking 기법 활용
### Note
* R 로 작성된 코드 같은 경우 파일을 분실하여 원본 코드를 보유하고 있지 않아 피피티 보고서로 대체

 

 
 


    


