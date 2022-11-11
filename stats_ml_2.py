# %%
# Python標準モジュール
from IPython.display import HTML
from pyvis.network import Network
import networkx as nx
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
import time

# %%
#
# csvファイルの読み込み
#

# ヘッダーの定義
header = ['Round', 'Hteam', 'Ateam', 'Hsocore', 'Ascore', 'HorA', 'WorL', 'Date', 'Wether',  'temperature', 'Grass_condition', 'Spectators', 'TimeLine1', 'TimeLine2', 'TimeLine3',
          'TimeLine4', 'TimeLine5', 'TimeLine6', 'Attack_CBP', 'Pass_CBP', 'Cross_CBP', 'Dribble_CBP', 'Shots_CBP', 'Score_CBP', 'Seizure', 'Diffence', 'Save_CBP', 'Expect_Goal',
          'Shots', 'Shots_Success', 'On_Target', 'PK', 'Pass', 'Pass_Success', 'Cross', 'Cross_Success', 'D_FK', 'I_FK', 'CK', 'Throwing', 'Throwing_Success', 'Dribble', 'Dribble_Success', 'Tackle', 'Tackle_Success', 'Clear', 'Intercept', 'OffSide', 'Yellow', 'Red', 'Approach_30m', 'Approach_Penalty', 'Attack_Num', 'Chances', 'Control']  # , 'AGI', 'KAGI'

# スタッツデータとAGIデータの読み込み
stats = pd.read_csv('/Users/okumaryo/Class/Laboratory/Twitter/J2_stats/kofu_stats_.csv')
agi = pd.read_csv('/Users/okumaryo/Class/Laboratory/Twitter/AGI_stats/kofu_stats_agi.csv')
agi.drop(columns='Round', inplace=True)
# スタッツとAGIの結合
stats = pd.concat([stats, agi], axis=1)
stats.loc[0]

# %%
# stats['HorA'].replace(['A', 'H'], [0, 1], inplace=True)
# 天候による影響度(改善の余地あり)
stats['Wether'].replace(['晴', '屋内', '晴時々曇', '晴のち曇', '晴一時曇', '曇時々晴', '曇のち晴', '曇一時晴', '晴時々雨', '晴のち雨', '晴一時雨', '曇', '曇時々雨', '曇のち雨', '曇一時雨', '雨時々晴', '雨のち晴', '雨一時晴', '雨時々曇', '雨のち曇', '雨一時曇', '雨', '曇のち雷雨時々晴', '曇時々雪', '雪', '曇時々雨一時雷'], [
                        1, 1, 1, 1, 1, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0, 0, 0.5, 0.5, 0], inplace=True)
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
stats.loc[0]

# %%
stats.describe()
# print(stats['Wether'].unique())
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
# 得点を結合
Score = pd.concat([HScore, AScore], axis=0)
Score.head()

# %%
stats = pd.merge(stats, Score, on='Round', how='outer')
stats.head()

# %%
stats.shape

# %%
# アウェイの場合の失点
ACons = pd.DataFrame(stats[stats['HorA'] == 0])
ACons = ACons.iloc[:, 0:1]
ACons.rename(columns={'Hscore': 'Cons'}, inplace=True)
ACons.head()
# %%
# ホームの場合の失点
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
stats['temperature'] = stats['temperature'].astype('float')
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
# 標準化の処理の実行（説明変数に対してのみ実行）
sc = StandardScaler()

# %%
# 欠損値の有無確認
stats.isnull().sum()
# %%
# 統計データ処理に不要な特徴の削除
drp = ['Hscore', 'Ascore', 'Shots_Success', 'WorL', 'Wether',
       'Grass_condition', 'Intercept', 'OffSide', 'Yellow', 'Red', 'Tackle_Success']
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
stats.loc[161, :]

# %%
# 時系列に訓練データと検証データの分割
train_x = sc_df_x[:162]
test_x = sc_df_x[:162].mean()
train_y = df_y[:162]
test_y = df_y[162]
test_x = pd.DataFrame(test_x)
test_x = test_x.T
test_y
# %%
# 訓練データとテストデータにスプリット
train_x, test_x, train_y, test_y = train_test_split(
    sc_df_x, df_y, test_size=0.3, random_state=2, stratify=df_y)
# train_x, test_x, train_y, test_y = sc_df_x[:115], sc_df_x[115:], df_y[:115], df_y[115:]
print("分割後のデータ")
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
print(np.bincount(df_y))

# %%
df_y.nunique()
# %%
# k近傍法
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(train_x, train_y)
knn_y_pred = knn.predict(test_x)
knn_y_pred[0]
# %%
# 性能評価
print("正解率: " + str(round(accuracy_score(test_y, knn_y_pred), 3)))
print("適合率: " + str(round(precision_score(test_y, knn_y_pred, average="macro"), 3)))
print("再現率: " + str(round(recall_score(test_y, knn_y_pred, average="macro"), 3)))

# %%
# 最適なkの値を調べる
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
# 特徴量のヒストグラム
for col in df_x:
    print(col)
    df_x[col].plot.hist()
    plt.show()
# %%
# 2変数間の散布図
plt.figure(figsize=(10, 10))
sns.scatterplot(data=stats, x="Attack_CBP", y="WorL")

# %%
stats['WorL'].nunique

# %%
# ヒートマップ(相関図)
plt.figure(figsize=(10, 10))
options = {'square': True, 'annot': True, 'fmt': '0.2f', 'xticklabels': sc_df_x.columns,
           'yticklabels': sc_df_x.columns, 'annot_kws': {'size': 5}, 'vmin': -1, 'vmax': 1, 'center': 0, 'cbar': False}
ax = sns.heatmap(sc_df_x.corr(), **options)
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)
# %%
# グラフネットワーク

threshold = 0.3
edge_width = 10

stats_corr = stats.corr()
mask_stats = stats_corr.mask(np.triu(np.ones(stats_corr.shape)).astype(
    bool), None)  # 「右上の三角行列」にマスクをして、Noneに置き換える
mask_stats
# %%
# edges data frame
edges = mask_stats.stack().reset_index().rename(
    columns={"level_0": "source", "level_1": "target", 0: "weight", })
edges
# %%
edges = edges.loc[abs(edges['weight']) > threshold]  # 該当のノードのみ出したい場合コメントアウト外す
edges['width'] = edges['weight'].apply(lambda x: abs(
    x) * edge_width if abs(x) > threshold else 0)  # weightに応じて、エッジの太さを変更
edges['color'] = edges['weight'].apply(
    lambda x: '#33A5CC' if x > threshold else '#BD4141')  # edgeの色
edges['label'] = edges['weight'].apply(lambda x: np.round(x, 2) if abs(x) > threshold else '')
edges
# %%
# networkxからpyvisに変換
G = nx.from_pandas_edgelist(edges, edge_attr=True)
nt = Network(height=f'1000px', width=f'1000px', bgcolor="#FFFFFF", font_color="black",
             notebook=True, directed=False)  # heading='test graph',
nt.from_nx(G)
nt
# グラフ構造
nt.repulsion(node_distance=300)
nt.edges
# %%
# ノードのレイアウト
neighbor_map = nt.get_adj_list()
neighbor_map

# %%
for n in nt.nodes:
    n['font'] = {'size': 20, 'strokeWidth': 6}
    n['size'] = 15  # shapeの内部にラベルがないノード形状のサイズ

# エッジのレイアウト
for e in nt.edges:
    if e['width'] >= threshold:
        e['font'] = {'size': 30, 'strokeWidth': 30, 'color': '#33A5CC'}  # strokeWidth: weightの背景範囲
    else:
        e['font'] = {'size': 30, 'strokeWidth': 10, 'color': '#BD4141'}

nt.show_buttons(filter_=['physics'])
nt.show("tmp.html")
display(HTML("tmp.html"))

# %%
# %%
