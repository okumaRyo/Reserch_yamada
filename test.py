# Python標準モジュール
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from catboost import Pool, CatBoostClassifier
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.linear_model import LogisticRegression
# 外部パッケージ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xgboost as xgb
import lightgbm as lgb  # LightGBM
# import optuna.integration.lightgbm as lgb
from multiprocessing import Process


corr_num, H_pre, A_pre, F_pre, game_num = 0, 0, 0, 0, 1
WDL = [[0], [0], [0]]
P_WDL = [[0], [0], [0]]
T_WDL = [[0], [0], [0]]
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


def stats_learning(team, stats_, games_num, k):
    sc = StandardScaler()
    stats = stats_.copy()
    # 天候の要素変換は要チェック
    # stats['WorL'].mask(stats['WorL'] != 0, 1, inplace=True)
    stats['Wether'].replace(['晴', '屋内', '晴時々曇', '晴のち曇', '晴一時曇', '曇時々晴', '曇のち晴', '曇一時晴', '晴時々雨',
                             '晴のち雨', '晴一時雨', '曇', '曇時々雨', '曇のち雨', '曇一時雨', '雨時々晴', '雨のち晴', '雨一時晴',
                             '雨時々曇', '雨のち曇', '雨一時曇', '雨', '曇のち雷雨時々晴', '曇時々雪', '雪', '曇時々雨一時雷',
                             '曇のち雨のち曇', '曇のち雷雨のち曇', '雪のち曇', '曇のち雨のち晴', '曇のち雷雨のち雨', '晴のち曇一時雨',
                             '曇時々雨のち曇', '曇一時雪', '晴のち曇時々雨', '晴時々曇一時雨', '雨のち雷雨のち雨', '雨時々雪',
                             '雨時々雷雨のち曇', '晴のち曇のち雨', '曇時々晴一時雨', '曇のち晴一時雨'], [
        1, 1, 1, 1, 1, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0.25, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0,
        0, 0, 0.25, 0.25, 0.25], inplace=True)
    stats['Grass_condition'].replace(['全面良芝', '良芝', '不良'], [1.0, 0.75, 0.6], inplace=True)
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
    stats.drop(columns=['Hteam', 'Ateam', 'Date'], inplace=True)

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
    stats['Shots_Success'] = stats['Shots_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Pass_Success'] = stats['Pass_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Cross_Success'] = stats['Cross_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Throwing_Success'] = stats['Throwing_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Dribble_Success'] = stats['Dribble_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Tackle_Success'] = stats['Tackle_Success'].apply(
        lambda x: x.replace('%', '')).astype('float')
    stats['Chances'] = stats['Chances'].apply(lambda x: x.replace('%', '')).astype('float')
    stats['Control'] = stats['Control'].apply(lambda x: x.replace('%', '')).astype('float')

    # 統計データ処理に不要な特徴の削除
    drp = ['Hscore', 'Ascore', 'Shots_Success', 'WorL']
    df_x = stats.drop(drp, axis=1)
    df_y = stats['WorL']
    true_y = stats_['WorL']
    true_y = true_y[games_num]

    """ svc = LinearSVC(C=0.01, penalty="l1", dual=False)
    selector = SelectFromModel(svc)

    X_new = pd.DataFrame(selector.fit_transform(df_x, stats['Survived']),
                         columns=df_x.columns.values[selector.get_support()])
    result = pd.DataFrame(selector.get_support(), index=df_x.columns.values,
                          columns=['False: dropped'])
    result['coef'] = selector.estimator_.coef_[0]  # Linearモデル用 """

    # df_xの標準化
    sc.fit(df_x)
    sc_df_x = pd.DataFrame(sc.transform(df_x))
    sc_df_x.columns = df_x.columns

    rf = RandomForestClassifier()
    # RFE(クロスバリデーションなし)Backwardと共通関数で、Defaultがforward
    """ selector = RFE(rf, n_features_to_select=15)

    X_new = pd.DataFrame(selector.fit_transform(sc_df_x, stats['WorL']),
                         columns=sc_df_x.columns.values[selector.get_support()]) """

    # RFECV(クロスバリデーションあり)
    """ in_features_to_select = 15
    selector = RFECV(rf, min_features_to_select=in_features_to_select, cv=5)

    X_new = pd.DataFrame(selector.fit_transform(sc_df_x, stats['WorL']),
                         columns=sc_df_x.columns.values[selector.get_support()]) """
    """ pca = PCA(n_components=40, random_state=0)
    umap = umap.UMAP(n_components=2, random_state=0)
    pca_x = pca.fit_transform(sc_df_x)
    X_reduced_umap = umap.fit_transform(pca_x) """

    # 時系列に訓練データと検証データの分割
    """ train_x = sc_df_x[:games_num]
    test_x = sc_df_x[games_num-5:games_num].mean()
    train_y = df_y[:games_num]
    test_y = df_y[games_num]
    test_x = pd.DataFrame(test_x)
    test_x = test_x.T """

    train_x = sc_df_x[:games_num]
    test_x_ = sc_df_x[games_num:]
    test_x = test_x_.copy()
    test_x[0:1] = sc_df_x[games_num-5:games_num].mean()
    train_y = df_y[:games_num]
    test_y = df_y[games_num:]
    # test_x = pd.DataFrame(test_x)
    # test_x = test_x.T

    # ロジスティック回帰
    if team == 2:
        model = LogisticRegression(penalty='l2',          # 正則化項(L1正則化 or L2正則化が選択可能)
                                   dual=False,            # Dual or primal
                                   tol=0.0001,            # 計算を停止するための基準値
                                   C=1.0,                 # 正則化の強さ
                                   fit_intercept=True,    # バイアス項の計算要否
                                   intercept_scaling=1,   # solver=‘liblinear’の際に有効なスケーリング基準値
                                   class_weight=None,     # クラスに付与された重み
                                   random_state=None,     # 乱数シード
                                   solver='lbfgs',        # ハイパーパラメータ探索アルゴリズム
                                   max_iter=100,          # 最大イテレーション数
                                   multi_class='auto',    # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                                   verbose=0,             # liblinearおよびlbfgsがsolverに指定されている場合、冗長性のためにverboseを任意の正の数に設定
                                   warm_start=False,      # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                                   n_jobs=None,           # 学習時に並列して動かすスレッドの数
                                   l1_ratio=None)
        model.fit(train_x, train_y)
        logi_pred = model.predict(test_x)
        return logi_pred[0], true_y, 'Logistic'

    # LinearSVC
    if team == 1:
        linsvc = LinearSVC(max_iter=10000)
        linsvc.fit(train_x, train_y)
        linsvc_pred = linsvc.predict(test_x)
        return linsvc_pred[0], true_y, 'LinearSVC'

    # LightGBM
    """ params = {
        'objective': 'binary',
        'metric': 'auc',
    }
    num_round = 100
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
    model = lgb.train(params, lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      early_stopping_rounds=50
                      ) """
    if team == 3:
        model = lgb.LGBMClassifier()
        model.fit(train_x, train_y)
        gbm_pred = model.predict(test_x)
        """ if gbm_pred[0] > 0.5:
            gbm_pred[0] = 1
        else:
            gbm_pred[0] = 0 """
        return gbm_pred[0], true_y, 'LightGBM'

    # k近傍法
    if team == 0:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x, train_y)
        knn_y_pred = knn.predict(test_x)
        return knn_y_pred[0], true_y, 'Kneighbors'

    """ # 勾配ブースティング
    train_pool = Pool(train_x, train_y)
    test_pool = Pool(test_x, test_y)
    params = {
        'early_stopping_rounds': 10,
        'iterations': 100,
        'custom_loss': ['Accuracy'],
        'random_seed': 42}
    model = CatBoostClassifier(**params)
    cab = model.fit(train_pool, eval_set=test_pool)
    preds = cab.predict(test_x)
    return preds[0], true_y """


def split_main(corr_num, H_pre, A_pre, F_pre, game_num, WDL_, P_WDL_, T_WDL_, start):
    # i = start
    for i in range(22):
        if i < 22:
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

                    H_pre, H_rslt, Alg = stats_learning(
                        start, Hteam_stats, game_start, 15)        # ホームチームの学習
                    A_pre, A_rslt, ppp = stats_learning(
                        start, Ateam_stats, A_game_start[0], 15)    # アウェイチームの学習

                    if H_pre == A_pre:
                        F_pre = 1
                        P_WDL[1][0] += 1
                    elif H_pre != 0 and A_pre != 3:
                        F_pre = 3
                        P_WDL[0][0] += 1
                    elif H_pre != 3 and A_pre != 0:
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
                    hgame += 1
                    #print(game_num, i, k, H_rslt, H_pre, A_pre)
        #i += 4
    print(f'multi_task = {start}\n{Alg}')
    print(
        f'正解率 = {corr_num} / {game_num}\n= {corr_num/game_num*100}%\n(Win, Draw, Lose) = {WDL}\npred(Win, Draw, Lose) = {P_WDL}\nTrue(Win, Draw, Lose) = {T_WDL}\n正答率 = W({WDL[0][0]/P_WDL[0][0]}) D({WDL[1][0]/P_WDL[1][0]}) L({WDL[2][0]/P_WDL[2][0]})\n正答率 = W({WDL[0][0]/T_WDL[0][0]}) D({WDL[1][0]/T_WDL[1][0]}) L({WDL[2][0]/T_WDL[2][0]})')

    # return 2, 4, np.array([[1], [2], [3]]), np.array([[10], [20], [30]]), np.array([[11], [21], [13]])


if __name__ == '__main__':
    k_ = 15
    max_corr = 0
    time_start = time.time()
    # while k_ < 99:
    for h in range(1):
        workers = [Process(target=split_main, args=(
            0, 0, 0, 0, 0, [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], i)) for i in range(4)]
        for work in workers:
            work.start()
        for worker in workers:
            worker.join()
            """ corr_num_, game_num_, WDL_, P_WDL_, T_WDL_ = work.start()
            corr_num += corr_num_
            game_num += game_num_
            WDL += WDL_
            P_WDL += P_WDL_
            T_WDL += T_WDL_ """

    time_end = time.time()
    print(f"{time_end - time_start}秒")
