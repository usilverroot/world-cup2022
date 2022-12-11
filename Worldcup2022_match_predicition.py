import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd


#국가대항전 데이터 받기
df = pd.read_csv('C:/Users/dldbs/OneDrive/바탕 화면/archive/international_matches.csv')
df = df.replace({'IR Iran': 'Iran','Korea Republic':'Korea'})
df = df.fillna(df.mean())

#데이터 정제하기(피파 랭킹 평균,피파 랭킹 차이, 평균 수비 스코어 차이, 평균 공격 스코어 차이, 매치 승)
df['rank_avg'] = (df['home_team_fifa_rank'] + df['away_team_fifa_rank'])/2
df['rank_diff'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['mean_defense_score_diff'] = df['home_team_goalkeeper_score'] + df['home_team_mean_defense_score'] - df['away_team_goalkeeper_score'] - df['away_team_mean_defense_score']
df['mean_offense_score_diff'] = df['home_team_mean_offense_score'] +  df['home_team_mean_midfield_score'] - df['away_team_mean_offense_score'] - df['away_team_mean_midfield_score']
df['match_win'] = (df['home_team_score'] - df['away_team_score'])>0

#데이터 학습 준비
X, y = df.loc[:,['rank_avg', 'rank_diff','mean_defense_score_diff', 'mean_offense_score_diff']], df['match_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


#Support Vector Machines
svc = svm.SVC(probability=True)
svc.fit(X_train, y_train)
svm_predict = svc.predict(X_test)

#2022월드컵팀 
worldcup2022_teams = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon', 'Canada', 'Costa Rica', 'Croatia', 'Denmark', 'Ecuador', 'England', 'France', 'Germany', 'Ghana', 
                 'Iran', 'Japan','Korea', 'Mexico', 'Morocco', 'Netherlands','Poland', 'Portugal', 'Qatar', 'Saudi Arabia', 'Senegal','Serbia', 'Spain', 'Switzerland',
                  'Tunisia','USA','Uruguay','Wales']

#2019년도 이후의 데이터 정제
worldcup_rankings_home = df[['home_team','home_team_fifa_rank']].loc[df['home_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')] 
worldcup_rankings_away = df[['away_team','away_team_fifa_rank']].loc[df['away_team'].isin(worldcup2022_teams)& (df['date']>'2019-01-01')]
worldcup_home_goalkeeper_score= df[['home_team','home_team_goalkeeper_score']].loc[df['home_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_home_mean_defense_score = df[['home_team', 'home_team_mean_defense_score']].loc[df['home_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_away_goalkeeper_score= df[['away_team','away_team_goalkeeper_score']].loc[df['away_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_away_mean_defense_score = df[['away_team', 'away_team_mean_defense_score']].loc[df['away_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_home_mean_offense_score= df[['home_team','home_team_mean_offense_score']].loc[df['home_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_home_mean_midfield_score= df[['home_team','home_team_mean_midfield_score']].loc[df['home_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_away_mean_midfield_score= df[['away_team','away_team_mean_midfield_score']].loc[df['away_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]
worldcup_away_mean_offense_score= df[['away_team','away_team_mean_offense_score']].loc[df['away_team'].isin(worldcup2022_teams) & (df['date']>'2019-01-01')]

#2019년도 이후의 데이터 평균값
worldcup_rankings_home = worldcup_rankings_home.groupby('home_team').mean()
worldcup_rankings_away = worldcup_rankings_away.groupby('away_team').mean()
worldcup_home_goalkeeper_score = worldcup_home_goalkeeper_score.groupby('home_team').mean()
worldcup_home_mean_defense_score = worldcup_home_mean_defense_score.groupby('home_team').mean()
worldcup_home_mean_offense_score = worldcup_home_mean_offense_score.groupby('home_team').mean()
worldcup_home_mean_midfield_score = worldcup_home_mean_midfield_score.groupby('home_team').mean()
worldcup_away_goalkeeper_score = worldcup_away_goalkeeper_score.groupby('away_team').mean()
worldcup_away_mean_defense_score = worldcup_away_mean_defense_score.groupby('away_team').mean()
worldcup_away_mean_midfield_score = worldcup_away_mean_midfield_score.groupby('away_team').mean()
worldcup_away_mean_offense_score = worldcup_away_mean_offense_score.groupby('away_team').mean()
worldcup_rankings_home['home_defense_score'] = worldcup_home_goalkeeper_score['home_team_goalkeeper_score']+worldcup_home_mean_defense_score['home_team_mean_defense_score']
worldcup_rankings_home['home_offense_score'] = worldcup_home_mean_offense_score['home_team_mean_offense_score']+worldcup_home_mean_midfield_score['home_team_mean_midfield_score']
worldcup_rankings_away['away_defense_score'] = worldcup_away_goalkeeper_score['away_team_goalkeeper_score']+worldcup_away_mean_defense_score['away_team_mean_defense_score']
worldcup_rankings_away['away_offense_score'] = worldcup_away_mean_offense_score['away_team_mean_offense_score']+worldcup_away_mean_midfield_score['away_team_mean_midfield_score']

#승부예측
def Match_Prediction(home,away):
    #데이터 정리하기
    match_predicition = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan]]), columns = X_train.columns)

    home_rank = worldcup_rankings_home.loc[home, 'home_team_fifa_rank']
    home_offense_score = worldcup_rankings_home.loc[home, 'home_offense_score']
    home_defense_score = worldcup_rankings_home.loc[home, 'home_defense_score']
    away_rank = worldcup_rankings_away.loc[away, 'away_team_fifa_rank']
    away_offense_score = worldcup_rankings_away.loc[away, 'away_offense_score']
    away_defense_score = worldcup_rankings_away.loc[away, 'away_defense_score']

    match_predicition['rank_avg'] = (home_rank + away_rank) / 2
    match_predicition['rank_diff'] = home_rank - away_rank
    match_predicition['mean_defense_score_diff'] = home_defense_score - away_defense_score
    match_predicition['mean_offense_score_diff'] = home_offense_score - away_offense_score
    
    #svc모델에 데이터 넣기
    home_win_prob = svc.predict_proba(match_predicition)[:,1][0]
    print("{} vs {}".format(home,away))
    print("{} is win with {:.2f} and lose with {:.2f}.".format(home,(home_win_prob)*100,(1- home_win_prob)*100))

#두개의 팀 입력받기     
if __name__ == '__main__':
    print("두 팀을 입력하세요.")
    while True:
        try:
            H,A = map(str, input().split())
            Match_Prediction(H,A)    
        except:   
            print("두 팀을 다시 입력하세요.")