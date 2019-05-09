# 2018-Big-Data-X-Campus
* Predict whether a person that has used Naeil Baeum card provided by the government has acquired a job and ultimately analyze the efficiency of such unemployment relief policy.
## Requirements
* R 3.5 (or later)
* RHINO (used instead of KoNLP)
## Packages
* ```library(glmnet)```
* ```library(caret)```
* ```library(xgboost)```
* ```library(nnet)```
* ```library(randomForest)```
* ```library(tree)```
* ```library(corrplot)```
## Methods for Training and Prediction
* ```Neural Network```
* ```Decision Tree```
* ```randomForest```
* ```C5.0```
* ```XGBoost```
* ```Linear Regression```
## How to use
### Data
* 한국고용정보원
* 정보공개포털
### Signature Table


![image](https://user-images.githubusercontent.com/42960718/53009220-60703200-347e-11e9-92f8-e926199d12a2.png)
### Data Pre-processing
* NA 처리
  * 원본데이터의 결측치를 포함한 칼럼 제거
    * 결측치가 발생한 데이터는 다른 값으로 대체 불가능하고 그 양이 데이터에 비해 적기 때문에 제거 결정
* 이상치 제거
  * 날짜를 잘못 입력한 휴먼 에러의 경우 → 휴먼 에러 3건 발견 후 삭제
  * 발급 후 곧바로 취업한 경우 → 담당자 인터뷰를 통한 domain knowledge 를 활용하여 곧바로 취업한 경우를 이상치 처리
* 희망직종분류
  * 교육프로그램명만 명시된 데이터를 받았기 때문에 통계청 산업대분류 데이터를 참고하여 산업별 분류 데이터 추가
* Correlation matrix


![image](https://user-images.githubusercontent.com/42960718/53009554-26536000-347f-11e9-9cef-e4df0976d4a1.png)
* Lasso 변수 선택법 



![image](https://user-images.githubusercontent.com/42960718/53009628-4aaf3c80-347f-11e9-8b13-9eaed802e90b.png)
  * 최적의 변수 13개 추출
    * sex
    * age
    * avgJobRate
    * spendDate
    * locationRatio
    * kospiPrice
    * diffKospiRatio
    * diffKosdaqRatio
    * 평균근속년수.년.
    * 총근로시간.시간.
    * 월임금총액.천원.
    * 근로자수.명.
### Model
* ```Linear Regression```
 * Accuracy 과적합 문제 및 데이터가 선형성을 띄지 않기 때문에 탈락
* Model comparison


![image](https://user-images.githubusercontent.com/42960718/53011926-e7280d80-3484-11e9-8877-368894ae8962.png)
#### Final Model
* 취업여부예측 : ```XGBoost```
* 취업준비기간 : ```randomForest```
### Search Engine 
* **pg.35 혹은 검색기 시연 영상.mp4 파일 참고**
* 내부 요소
  * 취업여부와 취업준비기간 예측 모델의 정확도를 검증하고 활용하기 위해 검색기 제작
  * 모델을 구동할 newdata 입력수치값을 찾는 것을 도와주는 역할
    * 나이
    * 성별
    * 희망 월임금
    * 희망직종
    * 지역관서
  * 입력시 해당되는 데이터셋 제시
* 외부 요소
  * 경제상황 지표인 코스피 지수 데이터셋 검색 가능
  * 카드발급날짜 입력시 해당 날짜 코스피 지수와 ```취직날짜 = 발급날짜``` 코스피지수 데이터셋 제시
* 검색기 예시


![image](https://user-images.githubusercontent.com/42960718/53012209-af6d9580-3485-11e9-9a12-177b7970d978.png)


  * 성별, 나이, 희망임금, 발급날짜 입력 → 희망직종 및 지역 센터명 입력 → 취업준비기간 일수와 취업 성공 여부 (0: 실패, 1: 성공) 제시
### Expectations
* 정부
  * 취업의지가 낮은 신청자를 선별하는 모델로 보완 가능
  * 불필요한 예산 낭비를 방지하여 더 많은 신청자 수용 가능
  * 발급자들의 수요를 기반으로 부족한 교육 프로그램 지원 확대
  * 취업할 것이라 예측했지만 실제로 취업하지 않은 경우 선별
    * 취업을 하지 못한 경우와 자기계발을 목적으로 카드를 악용하는 경우로 나뉘어짐
    * 후자의 경우 추적 조사 또는 심사 강화를 통하여 해당 사용자에게 패널티 부여 → 불필요한 예산 낭비 방지
* 개인
 * 교육과정 효과의 정량화를 통하여 내일배움카드제도의 효율성에 대한 불확실성 해소
 * 취준생에게 취업 준비에 필요한 최적의 인사이트를 제공
 * 개개인에게 맞춘 세분화된 취업 지원 서비스 제시
### Note
* R 로 작성된 코드 같은 경우 파일을 분실하여 원본 코드를 보유하고 있지 않아 피피티 보고서로 대체

