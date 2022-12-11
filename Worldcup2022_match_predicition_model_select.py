from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

#국가대항전 데이터 받기
df = pd.read_csv('C:/Users/dldbs/OneDrive/바탕 화면/archive/international_matches.csv')
df = df.replace({'IR Iran': 'Iran'})
df['date'] = pd.to_datetime(df['date'])
df = df.fillna(df.mean())

#데이터 정제하기
df['rank_avg'] = (df['home_team_fifa_rank'] + df['away_team_fifa_rank'])/2
df['rank_diff'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['mean_defense_score_diff'] = df['home_team_goalkeeper_score'] + df['home_team_mean_defense_score'] - df['away_team_goalkeeper_score'] - df['away_team_mean_defense_score']
df['mean_offense_score_diff'] = df['home_team_mean_offense_score'] +  df['home_team_mean_midfield_score'] - df['away_team_mean_offense_score'] - -df['away_team_mean_midfield_score']
df['match_win'] = (df['home_team_score'] - df['away_team_score'])>0

#데이터 학습 준비
X, y = df.loc[:,['rank_avg', 'rank_diff', 'mean_defense_score_diff', 'mean_offense_score_diff']], df['match_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


#GaussianNB로 트레이닝
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes_pred = naive_bayes.predict(X_test)
accuracy_naive_bayes = naive_bayes.score(X_test, y_test)
print(f'* Accuracy @ GaussianNB: {accuracy_naive_bayes:.3f}')

#KNN으로 트레이닝
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
Y_pred = KNN.predict(X_test)
accuracy_KNN = KNN.score(X_test, y_test)
print(f'* Accuracy @ KNN: {accuracy_KNN:.3f}')

#Support Vector Machines으로 트레이닝
svc =svm.SVC()
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_test)
accuracy_svc = svc.score(X_test, y_test)
print(f'* Accuracy @ Support Vector Machines: {accuracy_svc:.3f}')


#Decision Tree로 트레이닝
DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X_train, y_train)
Y_pred = DecisionTree.predict(X_test)
accuracy_DecisionTree = DecisionTree.score(X_test, y_test)
print(f'* Accuracy @ Decision Tree: {accuracy_DecisionTree:.3f}')

#Random Forest로 트레이닝
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)
Y_pred = RandomForest.predict(X_test)
RandomForest.score(X_train, y_train)
accuracy_RandomForest = RandomForest.score(X_test, y_test)
print(f'* Accuracy @ Random Forest: {accuracy_RandomForest:.3f}')

