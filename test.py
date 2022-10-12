# %%
# Python標準モジュール
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from operator import index
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import r2_score            # 決定係数
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from xml.dom.minicompat import defproperty

# 外部パッケージ
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

header = ['Round', 'Hteam', 'Ateam', 'Hsocore', 'Ascore', 'HorA', 'WorL', 'Date', 'Wether',  'temperature', 'Grass_condition', 'Spectators', 'TimeLine1', 'TimeLine2', 'TimeLine3',
          'TimeLine4', 'TimeLine5', 'TimeLine6', 'Attack_CBP', 'Pass_CBP', 'Cross_CBP', 'Dribble_CBP', 'Shots_CBP', 'Score_CBP', 'Seizure', 'Diffence', 'Save_CBP', 'Expect_Goal',
          'Shots', 'Shots_Success', 'On_Target', 'PK', 'Pass', 'Pass_Success', 'Cross', 'Cross_Success', 'D_FK', 'I_FK', 'CK', 'Throwing', 'Throwing_Success', 'Dribble', 'Dribble_Success', 'Tackle', 'Tackle_Success', 'Clear', 'Intercept', 'OffSide', 'Yellow', 'Red', 'Approach_30m', 'Approach_Penalty', 'Attack_Num', 'Chances', 'Control', 'AGI', 'KAGI']
stats = pd.read_csv('/Users/okumaryo/Class/Laboratory/Twitter/fcryukyu_stats_.csv')
#stats.set_index('Round', inplace=True)
# stats = stats.T
#stats.columns = header
stats.head()

# %%
# stats['HorA'].replace(['A', 'H'], [0, 1], inplace=True)
# 天候による影響度(改善の余地あり)
stats['Wether'].replace(['晴', '屋内', '晴時々曇', '晴のち曇', '晴一時曇', '曇時々晴', '曇のち晴', '曇一時晴', '晴時々雨', '晴のち雨', '晴一時雨', '曇', '曇時々雨', '曇のち雨', '曇一時雨', '雨時々晴', '雨のち晴', '雨一時晴', '雨時々曇', '雨のち曇', '雨一時曇', '雨', '曇のち雷雨時々晴'], [
                        1, 1, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.5, 0.5, 0.5, 0.75, 0.35, 0.35, 0.35, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0, 0.1], inplace=True)
stats['Grass_condition'].replace(['全面良芝', '良芝'], [1.0, 0.75], inplace=True)
stats['Date'] = stats['Date'].str.strip('Kick Off')
stats['Spectators'] = stats['Spectators'].str.strip('人')
stats['temperature'] = stats['temperature'].str.strip('℃')
stats['TimeLine1'] = stats['TimeLine1'].str.strip('%')
stats['TimeLine2'] = stats['TimeLine2'].str.strip('%')
stats['TimeLine3'] = stats['TimeLine3'].str.strip('%')
stats['TimeLine4'] = stats['TimeLine4'].str.strip('%')
stats['TimeLine5'] = stats['TimeLine5'].str.strip('%')
stats['TimeLine6'] = stats['TimeLine6'].str.strip('%')
stats['Shots_Success'] = stats['Shots_Success'].str.strip('%').str.strip('()')
stats['Pass_Success'] = stats['Pass_Success'].str.strip('%').str.strip('()')
stats['Cross_Success'] = stats['Cross_Success'].str.strip('()').str.strip('%')
stats['Throwing_Success'] = stats['Throwing_Success'].str.strip('()').str.strip('%')
stats['Dribble_Success'] = stats['Dribble_Success'].str.strip('()').str.strip('%')
stats['Tackle_Success'] = stats['Tackle_Success'].str.strip('()').str.strip('%')
stats['Chances'] = stats['Chances'].str.strip('%')
stats['Control'] = stats['Control'].str.strip('%')
stats['Spectators'] = stats['Spectators'].str.strip(',')
stats.drop(columns='Round', inplace=True)
stats.head()

# %%
stats.drop(columns=['Hteam', 'Ateam', 'Date'], inplace=True)

# %%
stats.head()

# %%
stats.shape

# %%
# ホームの場合の得点
HScore = pd.DataFrame(stats[stats['HorA'] == 1])
HScore = HScore.iloc[:, 0:1]
HScore.rename(columns={'Hscore': 'Score'}, inplace=True)
HScore.head()
# %%
# アウェイの場合の得点
AScore = pd.DataFrame(stats[stats['HorA'] == 0])
AScore = AScore.iloc[:, 0:2]
AScore.drop(columns='Hscore', inplace=True)
AScore.rename(columns={'Ascore': 'Score'}, inplace=True)
AScore.head()

# %%
Score = pd.concat([HScore, AScore], axis=0)
Score.head()

# %%
stats = pd.merge(stats, Score, on='Round', how='outer')
stats.head()

# %%
stats.shape

# %%
# ホームの場合の得点
ACons = pd.DataFrame(stats[stats['HorA'] == 0])
ACons = ACons.iloc[:, 0:1]
ACons.rename(columns={'Hscore': 'Cons'}, inplace=True)
ACons.head()
# %%
# アウェイの場合の得点
HCons = pd.DataFrame(stats[stats['HorA'] == 1])
HCons = HCons.iloc[:, [0, 1]]
HCons.drop(columns='Hscore', inplace=True)
HCons.rename(columns={'Ascore': 'Cons'}, inplace=True)
HCons.head()

# %%
Cons = pd.concat([HCons, ACons], axis=0)
Cons.head()

# %%
stats = pd.merge(stats, Cons, on='Round', how='outer')
stats.head()
# %%
stats.dtypes

# %%
# 型変換
stats['Spectators'] = stats['Spectators'].apply(lambda x: x.replace(',', '')).astype('int')

stats['Spectators'].head()

# %%
stats['temperature'] = stats['temperature'].astype('float')

# %%
stats.head()
# %%
stats['TimeLine1'] = stats['TimeLine1'].apply(lambda x: x.replace('%', '')).astype('float')
stats['TimeLine2'] = stats['TimeLine2'].apply(lambda x: x.replace('%', '')).astype('float')
stats['TimeLine3'] = stats['TimeLine3'].apply(lambda x: x.replace('%', '')).astype('float')
stats['TimeLine4'] = stats['TimeLine4'].apply(lambda x: x.replace('%', '')).astype('float')
stats['TimeLine5'] = stats['TimeLine5'].apply(lambda x: x.replace('%', '')).astype('float')
stats['TimeLine6'] = stats['TimeLine6'].apply(lambda x: x.replace('%', '')).astype('float')
stats['Shots_Success'] = stats['Shots_Success'].apply(lambda x: x.replace('%', '')).astype('float')
stats['Pass_Success'] = stats['Pass_Success'].apply(lambda x: x.replace('%', '')).astype('float')
stats['Cross_Success'] = stats['Cross_Success'].apply(lambda x: x.replace('%', '')).astype('float')
stats['Throwing_Success'] = stats['Throwing_Success'].apply(
    lambda x: x.replace('%', '')).astype('float')
stats['Dribble_Success'] = stats['Dribble_Success'].apply(
    lambda x: x.replace('%', '')).astype('float')
stats['Tackle_Success'] = stats['Tackle_Success'].apply(
    lambda x: x.replace('%', '')).astype('float')
stats['Chances'] = stats['Chances'].apply(lambda x: x.replace('%', '')).astype('float')
stats['Control'] = stats['Control'].apply(lambda x: x.replace('%', '')).astype('float')

# %%
# ライブラリーのインポート

# 標準化の処理の実行（説明変数に対してのみ実行）
sc = StandardScaler()

# %%
sc.fit(stats.iloc[:, :-1])
sc_stats = pd.DataFrame(sc.transform(stats.iloc[:, :-1]))
sc_stats.head()

# %%
sc_stats["Cons"] = stats["Cons"]
sc_stats.columns = stats.columns
sc_stats.head()
# %%
# データの分割

sc_X = sc_stats.iloc[:, :-1].values
sc_y = sc_stats.iloc[:, -1].values
# 再現性や後の検討を考えてrandom_stateは指定しておくことがおすすめ
sc_X_train, sc_X_test, sc_y_train, sc_y_test = train_test_split(
    sc_X, sc_y, test_size=0.3, random_state=1)

# データの分割を確認
print("分割後のデータ")
print(sc_X.shape)
print(sc_X_train.shape)
print(sc_X_test.shape)

# %%
reg_lr = LinearRegression()
# 訓練データにモデルを適用する
reg_lr.fit(sc_X_train, sc_y_train)


sc_stats.shape

# %%
sc_stats

# %%
stats.isnull().sum()
# %%
drp = ['Hscore', 'Ascore', 'WorL']
df_x = stats.drop(drp, axis=1)
df_y = stats['WorL']
df_x.head()
# %%
# df_xの標準化
sc.fit(df_x)
sc_df_x = pd.DataFrame(sc.transform(df_x))
sc_df_x.columns = df_x.columns
sc_df_x.head()
# %%
train_x, test_x, train_y, test_y = train_test_split(sc_df_x, df_y, test_size=0.3, random_state=2)
print("分割後のデータ")
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# %%
df_coef = pd.DataFrame(model.coef_)
df_coef
# %%
#model = linear_model.Lasso(alpha=0.1)
model = svm.LinearSVC(max_iter=10000, random_state=83)

model.fit(train_x, train_y)
pred_train = model.predict(train_x)
pred_test = model.predict(test_x)

# %%
df_coef = pd.DataFrame(model.coef_)
df_coef.index = train_x.columns
df_coef.style.bar()

# %%
rmse = np.sqrt(mean_squared_error(test_y, pred_test))
rmse
# %%
stats

# %%
df_x = stats.drop(drp, axis=1)
df_x

# %%
print('R^2 学習: %.2f, テスト: %.2f' % (
    r2_score(train_y, pred_train),  # 学習
    r2_score(test_y, pred_test)    # テスト
))
# %%
stats.describe()
print(classification_report(test_y, pred_test, target_names=['0', '1'], digits=4))

# %%
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(train_x, train_y)

# %%
knn_y_pred = knn.predict(test_x)
# %%
# 性能評価
print("正解率: " + str(round(accuracy_score(test_y, knn_y_pred), 3)))
print("適合率: " + str(round(precision_score(test_y, knn_y_pred, average="macro"), 3)))
print("再現率: " + str(round(recall_score(test_y, knn_y_pred, average="macro"), 3)))

# %%

accuracy = []
precision = []
recall = []

k_range = range(1, 100)

for k in k_range:

    # モデルインスタンス作成
    knn = KNeighborsClassifier(n_neighbors=k)

    # モデル学習
    knn.fit(train_x, train_y)

    # 性能評価
    knn_y_pred = knn.predict(test_x)
    accuracy.append(round(accuracy_score(test_y, knn_y_pred), 3))
    precision.append(round(precision_score(test_y, knn_y_pred, average="macro"), 3))
    recall.append(round(recall_score(test_y, knn_y_pred, average="macro"), 3))

# グラフプロット
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# 結果出力
max_accuracy = max(accuracy)
index = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("「k="+str(best_k_range)+"」の時、正解率は最大値「"+str(max_accuracy)+"」をとる")

# %%
df_x
# %%
