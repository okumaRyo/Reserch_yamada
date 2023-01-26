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
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, RFE, RFECV
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from catboost import Pool, CatBoostClassifier
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import connected_components
from umap import UMAP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
# 外部パッケージ
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import lightgbm as lgb  # LightGBM
import optuna
from multiprocessing import Process
import os

#
# {チーム名：チーム略称名}の辞書型リストの作成
#
j2_team = {'いわてグルージャ盛岡': 'iwte', 'ベガルタ仙台': 'send', 'ブラウブリッツ秋田': 'aki', 'モンテディオ山形': 'yama', '水戸ホーリーホック': 'mito',
           '栃木ＳＣ': 'toch', 'ザスパクサツ群馬': 'gun', '大宮アルディージャ': 'omiy', 'ジェフユナイテッド千葉': 'chib', '東京ヴェルディ': 'tk-v',
           'ＦＣ町田ゼルビア': 'mcd', '横浜ＦＣ': 'y-fc', 'ヴァンフォーレ甲府': 'kofu', 'アルビレックス新潟': 'niig', 'ツエーゲン金沢': 'kana',
           'ファジアーノ岡山': 'okay', 'レノファ山口ＦＣ': 'r-ya', '徳島ヴォルティス': 'toku', 'Ｖ・ファーレン長崎': 'ngsk', 'ロアッソ熊本': 'kuma',
           '大分トリニータ': 'oita', 'ＦＣ琉球': 'ryuk'}
j2_Hteams = ['iwte', 'send', 'aki', 'yama', 'mito', 'toch', 'gun', 'omiy', 'chib', 'tk-v',
             'mcd', 'y-fc', 'kofu', 'niig', 'kana', 'okay', 'r-ya', 'toku', 'ngsk', 'kuma', 'oita', 'ryuk']


class LightGBM:
    def __init__(self, stats_, games_num):
        sc = StandardScaler()
        self.games_num = games_num
        self.stats = stats_.copy()
        self.stats_ = stats_.copy()
        # 天候の要素変換は要チェック
        self.stats['WorL'].mask(self.stats['WorL'] == 3, 2, inplace=True)
        self.stats['Wether'].replace(['晴', '屋内', '晴時々曇', '晴のち曇', '晴一時曇', '曇時々晴', '曇のち晴', '曇一時晴', '晴時々雨',
                                      '晴のち雨', '晴一時雨', '曇', '曇時々雨', '曇のち雨', '曇一時雨', '雨時々晴', '雨のち晴', '雨一時晴',
                                      '雨時々曇', '雨のち曇', '雨一時曇', '雨', '曇のち雷雨時々晴', '曇時々雪', '雪', '曇時々雨一時雷',
                                      '曇のち雨のち曇', '曇のち雷雨のち曇', '雪のち曇', '曇のち雨のち晴', '曇のち雷雨のち雨', '晴のち曇一時雨',
                                      '曇時々雨のち曇', '曇一時雪', '晴のち曇時々雨', '晴時々曇一時雨', '雨のち雷雨のち雨', '雨時々雪',
                                      '雨時々雷雨のち曇', '晴のち曇のち雨', '曇時々晴一時雨', '曇のち晴一時雨'], [
            1, 1, 1, 1, 1, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0.25, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0,
            0, 0, 0.25, 0.25, 0.25], inplace=True)
        self.stats['Grass_condition'].replace(['全面良芝', '良芝', '不良'], [1.0, 0.75, 0.6], inplace=True)
        self.stats['Date'] = self.stats['Date'].str.strip('Kick Off')
        self.stats['Spectators'] = self.stats['Spectators'].str.strip('人')
        self.stats['temperature'] = self.stats['temperature'].str.strip('℃')
        self.stats['TimeLine1'] = self.stats['TimeLine1'].str.strip('%')
        self.stats['TimeLine2'] = self.stats['TimeLine2'].str.strip('%')
        self.stats['TimeLine3'] = self.stats['TimeLine3'].str.strip('%')
        self.stats['TimeLine4'] = self.stats['TimeLine4'].str.strip('%')
        self.stats['TimeLine5'] = self.stats['TimeLine5'].str.strip('%')
        self.stats['TimeLine6'] = self.stats['TimeLine6'].str.strip('%')
        self.stats['Shots_Success'] = self.stats['Shots_Success'].str.strip('%').str.strip('()')
        self.stats['Pass_Success'] = self.stats['Pass_Success'].str.strip('%').str.strip('()')
        self.stats['Cross_Success'] = self.stats['Cross_Success'].str.strip('()').str.strip('%')
        self.stats['Throwing_Success'] = self.stats['Throwing_Success'].str.strip(
            '()').str.strip('%')
        self.stats['Dribble_Success'] = self.stats['Dribble_Success'].str.strip('()').str.strip('%')
        self.stats['Tackle_Success'] = self.stats['Tackle_Success'].str.strip('()').str.strip('%')
        self.stats['Chances'] = self.stats['Chances'].str.strip('%')
        self.stats['Control'] = self.stats['Control'].str.strip('%')
        self.stats['Spectators'] = self.stats['Spectators'].str.strip(',')
        self.stats.drop(columns='Round', inplace=True)
        self.stats.drop(columns=['Hteam', 'Ateam', 'Date'], inplace=True)

        # 型変換
        self.stats['Spectators'] = self.stats['Spectators'].apply(
            lambda x: x.replace(',', '')).astype('int')
        self.stats['Spectators'].head()
        self.stats['temperature'] = self.stats['temperature'].astype('float')
        self.stats['TimeLine1'] = self.stats['TimeLine1'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['TimeLine2'] = self.stats['TimeLine2'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['TimeLine3'] = self.stats['TimeLine3'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['TimeLine4'] = self.stats['TimeLine4'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['TimeLine5'] = self.stats['TimeLine5'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['TimeLine6'] = self.stats['TimeLine6'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Shots_Success'] = self.stats['Shots_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Pass_Success'] = self.stats['Pass_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Cross_Success'] = self.stats['Cross_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Throwing_Success'] = self.stats['Throwing_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Dribble_Success'] = self.stats['Dribble_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Tackle_Success'] = self.stats['Tackle_Success'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Chances'] = self.stats['Chances'].apply(
            lambda x: x.replace('%', '')).astype('float')
        self.stats['Control'] = self.stats['Control'].apply(
            lambda x: x.replace('%', '')).astype('float')

        # 統計データ処理に不要な特徴の削除
        drp = ['Hscore', 'Ascore', 'Shots_Success', 'WorL', 'Wether', 'Grass_condition']
        self.df_x = self.stats.drop(drp, axis=1)
        self.df_y = self.stats['WorL']
        self.true_y = stats_['WorL']
        self.true_y = self.true_y[self.games_num]

        # df_xの標準化
        sc.fit(self.df_x)
        self.sc_df_x = pd.DataFrame(sc.transform(self.df_x))
        self.sc_df_x.columns = self.df_x.columns

        # 時系列に訓練データと検証データの分割
        """ self.train_x = self.sc_df_x[:self.games_num]
        self.test_x = self.sc_df_x[self.games_num-5:self.games_num].mean()
        self.train_y = self.df_y[:self.games_num]
        self.test_y = self.df_y[self.games_num]
        self.test_x = pd.DataFrame(self.test_x)
        self.test_x = self.test_x.T """
        self.train_x = self.sc_df_x[:self.games_num]
        self.test_x_ = self.sc_df_x[self.games_num:]
        self.test_x = self.test_x_.copy()
        self.test_x[0:1] = self.sc_df_x[self.games_num-5:self.games_num].mean()
        self.train_y = self.df_y[:self.games_num]
        self.test_y = self.df_y[self.games_num:]

    def __call__(self, trial: optuna.Trial):
        import optuna.integration.lightgbm as lgb_opt
        params = {
            # 最適化アルゴリズムを指定

            'boosting': 'gbdt',                 # ブースティングの選択。 gbdtは勾配アンサンブル学習決定木 他は rf,dart,goss.
            'metric': 'multi_error',            # モデル構築の損失関数の指定。multi_errorは正解率
            'objective': 'multiclass',          # 予測の種類 regression binary など
            'num_class': 3,                     # クラス数
            'learning_rate': 0.005,              # 学習率 0.01 ~ 0.005 が主流
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),                 # L1正則化
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),                 # L2正則化
            # 1つの決定木に付く葉の数。学習するデータをどれだけ細かく分けるか。
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),
            # 特徴量を何%使うか。木を作る毎に、特徴量を指定した％分、ランダムに抽出し学習に使う。(列方向)
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            # 特徴量を何%使うか。木を作る毎に、特徴量を指定した％分、ランダムに抽出し学習に使う。(行方向)
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            # baggingする間隔。5を指定すると、学習5回するたびに、データをランダムに選び直す。
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
            # 一つの葉に入る最小データ数。デフォルトは20。
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'seed': 42,                         # ランダムシード値
            'verbosity': -1,                    # 学習途中の中継的な情報を表示するかしないか。-1は非表示。
        }

        self.lgb_train = lgb_opt.Dataset(self.train_x, self.train_y)
        self.lgb_eval = lgb_opt.Dataset(self.test_x, self.test_y, reference=self.lgb_train)

        callbacks = [
            lgb.log_evaluation(-1),
            lgb.early_stopping(100),            # 指定回数精度の向上がなければ打ち切り。
        ]
        evaluation_results = {}
        model = lgb_opt.train(params,
                              self.lgb_train,
                              valid_sets=[self.lgb_train, self.lgb_eval],
                              valid_names=["Train", "Test"],
                              evals_result=evaluation_results,          # metricの履歴の保存
                              # 生成する決定木の数(boosting回数)。early_stoppingで止めるため多めに設定。
                              num_boost_round=100000,
                              callbacks=callbacks,
                              )

        y_prob_val = model.predict(self.test_x, num_iteration=model.best_iteration)
        y_pred_val = np.argmax(y_prob_val, axis=1)

        score = accuracy_score(self.test_y, y_pred_val)
        return score

    def Logistic(self, bestparams):
        bestparams["objective"] = "multiclass"
        bestparams["metrics"] = "multi_error"
        bestparams["num_class"] = 3

        lgb_train = lgb.Dataset(self.train_x, self.train_y)
        lgb_eval = lgb.Dataset(self.test_x, self.test_y, reference=lgb_train)
        callbacks = [
            lgb.log_evaluation(-1),
            lgb.early_stopping(100),
        ]
        evaluation_results = {}
        model = lgb.train(bestparams,
                          lgb_train,
                          valid_sets=[lgb_train, lgb_eval],
                          valid_names=["Train", "Test"],
                          evals_result=evaluation_results,
                          num_boost_round=100000,
                          callbacks=callbacks,
                          )

        y_prod_test = model.predict(self.test_x,  num_iteration=model.best_iteration)
        y_pred_test = np.argmax(y_prod_test, axis=1)

        """ threshold = 0.5
        if y_pred_test[0] > threshold:
            y_pred_test[0] = 1
        else:
            y_pred_test[0] = 0 """
        return y_pred_test[0], self.true_y


def split_main(corr_num, H_pre, A_pre, F_pre, game_num, WDL, P_WDL, T_WDL, start):
    i = start
    while i < 22:
        # スタッツとAGIの取得
        stats = pd.read_csv(
            f'/Users/okumaryo/Class/Laboratory/Twitter/J2_stats/{j2_Hteams[i]}_stats_.csv')
        agi = pd.read_csv(
            f'/Users/okumaryo/Class/Laboratory/Twitter/AGI_stats/{j2_Hteams[i]}_stats_agi.csv')
        agi.drop(columns='Round', inplace=True)
        # スタッツとAGIの結合
        Hteam_stats = pd.concat([stats, agi], axis=1)
        hgame = 1
        for k in range(42):                 # 2022年の試合数
            game_start = k + len(Hteam_stats['HorA']) - 42      # 試合開始地点の特定
            if Hteam_stats.loc[game_start, 'HorA'] == 1:        # ホームでの試合の場合，以前の試合を学習
                A_stats = pd.read_csv(
                    f'/Users/okumaryo/Class/Laboratory/Twitter/J2_stats/{j2_team[Hteam_stats.loc[game_start, "Ateam"]]}_stats_.csv')
                A_agi = pd.read_csv(
                    f'/Users/okumaryo/Class/Laboratory/Twitter/AGI_stats/{j2_team[Hteam_stats.loc[game_start, "Ateam"]]}_stats_agi.csv')
                A_agi.drop(columns='Round', inplace=True)
                Ateam_stats = pd.concat([A_stats, A_agi], axis=1)
                game_date = Hteam_stats.loc[game_start, 'Date']
                A_game_start = Ateam_stats[Ateam_stats['Date'] == f'{game_date}'].index
                h_lgbm = LightGBM(Hteam_stats, game_start)
                a_lgbm = LightGBM(Ateam_stats, A_game_start[0])
                if hgame == 1:
                    h_study = optuna.create_study(
                        storage='mysql://lgbm_optuna:lgbm_lab@localhost/lightGBM_Lab', study_name='lightgbm_lab', direction='maximize', load_if_exists=True)
                    h_study.optimize(h_lgbm, n_trials=1)
                    hgame = 0
                a_study = optuna.create_study(
                    storage='mysql://lgbm_optuna:lgbm_lab@localhost/lightGBM_Lab', study_name='lightgbm_lab', direction='maximize', load_if_exists=True)
                a_study.optimize(a_lgbm, n_trials=1)

                # ベストパラメータを出力
                print('H_params:', h_study.best_params)
                print('A_params:', a_study.best_params)

                H_pre, H_rslt = h_lgbm.Logistic(h_study.best_params)
                A_pre, A_rslt = a_lgbm.Logistic(a_study.best_params)

                if H_rslt == 2:
                    H_rslt = 3

                # 最終予測
                if H_pre == A_pre:
                    F_pre = 1
                    P_WDL[1][0] += 1
                elif H_pre != 0 and A_pre == 0:
                    F_pre = 3
                    P_WDL[0][0] += 1
                elif H_pre == 0 and A_pre != 0:
                    F_pre = 0
                    P_WDL[2][0] += 1
                else:
                    F_pre = 1
                    P_WDL[1][0] += 1

                # 最終予測が正解している数
                if F_pre == H_rslt:
                    corr_num += 1

                if F_pre == 0 and H_rslt == 0:
                    WDL[2][0] += 1
                elif F_pre == 1 and H_rslt == 1:
                    WDL[1][0] += 1
                elif F_pre == 3 and H_rslt == 3:
                    WDL[0][0] += 1

                if H_rslt == 0:
                    T_WDL[2][0] += 1
                elif H_rslt == 1:
                    T_WDL[1][0] += 1
                elif H_rslt == 3:
                    T_WDL[0][0] += 1
                game_num += 1
                print(game_num, i, k, H_rslt, H_pre, A_pre)
        i += 2
    print(
        f'正解率 = {corr_num} / {game_num}\n= {corr_num/game_num*100}%\n(Win, Draw, Lose) = {WDL}\npred(Win, Draw, Lose) = {P_WDL}\nTrue(Win, Draw, Lose) = {T_WDL}\n正答率 = W({WDL[0][0]/P_WDL[0][0]}) D({WDL[1][0]/P_WDL[1][0]}) L({WDL[2][0]/P_WDL[2][0]})\n正答率 = W({WDL[0][0]/T_WDL[0][0]}) D({WDL[1][0]/T_WDL[1][0]}) L({WDL[2][0]/T_WDL[2][0]})')
    f = open('lightgbm.txt', 'a')
    f.write(
        f'multi = {i}\n正解率 = {corr_num} / {game_num}\n= {corr_num/game_num*100}%\n(Win, Draw, Lose) = {WDL}\npred(Win, Draw, Lose) = {P_WDL}\nTrue(Win, Draw, Lose) = {T_WDL}\n正答率 = W({WDL[0][0]/P_WDL[0][0]}) D({WDL[1][0]/P_WDL[1][0]}) L({WDL[2][0]/P_WDL[2][0]})\n正答率 = W({WDL[0][0]/T_WDL[0][0]}) D({WDL[1][0]/T_WDL[1][0]}) L({WDL[2][0]/T_WDL[2][0]})\n')
    f.close()


if __name__ == '__main__':
    corr_num, H_pre, A_pre, F_pre, game_num = 0, 0, 0, 0, 0
    WDL = [[0], [0], [0]]
    P_WDL = [[0], [0], [0]]
    T_WDL = [[0], [0], [0]]
    k_ = 23
    time_start = time.time()
    workers = [Process(target=split_main, args=(
        0, 0, 0, 0, 0, WDL, P_WDL, T_WDL, i)) for i in range(2)]
    for work in workers:
        work.start()
    for worker in workers:
        worker.join()

    """ print(
        f'正解率 = {corr_num} / {game_num}\n= {corr_num/game_num*100}%\n(Win, Draw, Lose) = {WDL}\npred(Win, Draw, Lose) = {P_WDL}\nTrue(Win, Draw, Lose) = {T_WDL}\n正答率 = W({WDL[0][0]/P_WDL[0][0]}) D({WDL[1][0]/P_WDL[1][0]}) L({WDL[2][0]/P_WDL[2][0]})\n正答率 = W({WDL[0][0]/T_WDL[0][0]}) D({WDL[1][0]/T_WDL[1][0]}) L({WDL[2][0]/T_WDL[2][0]})')
 """
    time_end = time.time()
    print(f"{time_end - time_start}秒")
