# World-cup2022 Match Predicition
## 프로젝트 제작 이유

2022년 여름이 아닌 겨울에 월드컵 시즌이 왔다. 평소에 축구를 자주 챙겨보는 학생으로서 국가대항전 축구 **승부 예측**은 항상 재미있다. 따라서 이번 월드컵에 진출한 32개의 팀에 한정하여 승부 예측 프로그램을 만들어보기로 했다.
## 프로그램 제작 과정

1. 승부 예측 프로그램을 위한 국가대항전 결과를 담은 데이터를 kaggle을 통해 구했다.
2. 승부 예측 모델에 입력할 수있도록 국가대항전 데이터를 정제했다.
* 데이터 셋은 *date, home_team, away_team, home_team_continent, away_team_continent, home_team_fifa_rank, away_team_fifa_rank, home_team_total_fifa_points, away_team_total_fifa_points, home_team_score, away_team_score, tournament, city, country, neutral_location, shoot_out, home_team_result, home_team_goalkeeper_score, away_team_goalkeeper_score, home_team_mean_defense_score, home_team_mean_offense_score, home_team_mean_midfield_score, away_team_mean_defense_score, away_team_mean_offense_score, away_team_mean_midfield_score*으로 구성되어있고 
이중에서 *date, home_team, away_team,home_team_fifa_rank, away_team_fifa_rank, home_team_score, away_team_score,home_team_goalkeeper_score, away_team_goalkeeper_score, home_team_mean_defense_score, home_team_mean_offense_score, home_team_mean_midfield_score, away_team_mean_defense_score, away_team_mean_offense_score, away_team_mean_midfield_score*
데이터로 이용하기로 결정했다. 

* 승부 예측 모델에 직접적으로 사용할 데이터는 피파랭킹 평균(홈팀,어웨이팀), 피파랭킹 차이(홈팀,어웨이팀), 평균 수비력 차이(홈팀,어웨이팀), 평균 공격력 차이(홈팀,어웨이팀)이다.

3.승부 예측 모델로 사용될 classifiers를 선정하고 데이터를 학습시켰다.
* ml_tutorial 30페이지에서 나오는 classifiers중에 GaussianNB, KNN, Support Vector Machines, Decision Tree, Random Forest 를 선정하여 각각 트레이닝한다.
* ![화면 캡처 2022-12-11 021131](https://user-images.githubusercontent.com/113349635/206885664-89b5f61f-52b3-4e3e-8ac2-48c7d3ff3a91.png)
* 가장 정확도가 높은 SVM을 모델로 선정했다.

4.2022년 월드컵에 진출한 32개의 팀의 데이터를 2019년도 이후의 데이터로 선정한다.
* 2018년도에 월드컵이 있었기 때문에 이와 같이 기준을 정했다.

5. 승부 예측을 구현헀다.

## 아쉬운점
* 승부 예측에 필요한 변수들이 충분하지 못하다고 생각한다.
* 무승부 확률은 구현해내지 못하였다.
* 골키퍼, 평균 수비수 * 미드필더 * 공격수 스코어의 NaN 데이터를 평균값으로 계산하여 정확도가 조금 낮아졌다.

## 참고
https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022
