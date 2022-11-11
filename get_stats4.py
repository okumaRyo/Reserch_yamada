# %%
import pandas as pd
import csv
import time
from numpy import packbits
from selenium import webdriver
import chromedriver_binary                  # 起動時バージョン確認
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


class GetStats:
    def __init__(self):
        self.header = ['Round', 'Hteam', 'Ateam', 'Hscore', 'Ascore', 'HorA', 'WorL', 'Date', 'Wether', 'Grass_condition', 'Spectators', 'temperature', 'TimeLine1', 'TimeLine2', 'TimeLine3',
                       'TimeLine4', 'TimeLine5', 'TimeLine6', 'Attack_CBP', 'Pass_CBP', 'Cross_CBP', 'Dribble_CBP', 'Shots_CBP', 'Score_CBP', 'Seizure', 'Diffence', 'Save_CBP', 'Expect_Goal',
                       'Shots', 'Shots_Success', 'On_Target', 'PK', 'Pass', 'Pass_Success', 'Cross', 'Cross_Success', 'D_FK', 'I_FK', 'CK', 'Throwing', 'Throwing_Success', 'Dribble', 'Dribble_Success', 'Tackle', 'Tackle_Success', 'Clear', 'Intercept', 'OffSide', 'Yellow', 'Red', 'Approach_30m', 'Approach_Penalty', 'Attack_Num', 'Chances', 'Control']  # 'AGI', 'KAGI'の追加も
        self.header_agi = ['Round', 'AGI', 'KAGI']
        self.url_vegalta = ['https://www.football-lab.jp/send/match/?year=2019', 'https://www.football-lab.jp/send/match/?year=2020',
                            'https://www.football-lab.jp/send/match/?year=2021', 'https://www.football-lab.jp/send/match/']
        self.url_ryukyu = ['https://www.football-lab.jp/ryuk/match/?year=2019', 'https://www.football-lab.jp/ryuk/match/?year=2020',
                           'https://www.football-lab.jp/ryuk/match/?year=2021', 'https://www.football-lab.jp/ryuk/match/']
        self.url_iwate = ['https://www.football-lab.jp/iwte/match/?year=2019', 'https://www.football-lab.jp/iwte/match/?year=2020',
                          'https://www.football-lab.jp/iwte/match/?year=2021', 'https://www.football-lab.jp/iwte/match/']
        self.url_akita = ['https://www.football-lab.jp/aki/match/?year=2019', 'https://www.football-lab.jp/aki/match/?year=2020',
                          'https://www.football-lab.jp/aki/match/?year=2021', 'https://www.football-lab.jp/aki/match/']
        self.url_yamagata = ['https://www.football-lab.jp/yama/match/?year=2019', 'https://www.football-lab.jp/yama/match/?year=2020',
                             'https://www.football-lab.jp/yama/match/?year=2021', 'https://www.football-lab.jp/yama/match/']
        self.url_mito = ['https://www.football-lab.jp/mito/match/?year=2019', 'https://www.football-lab.jp/mito/match/?year=2020',
                         'https://www.football-lab.jp/mito/match/?year=2021', 'https://www.football-lab.jp/mito/match/']
        self.url_tochigi = ['https://www.football-lab.jp/toch/match/?year=2019', 'https://www.football-lab.jp/toch/match/?year=2020',
                            'https://www.football-lab.jp/toch/match/?year=2021', 'https://www.football-lab.jp/toch/match/']
        self.url_gunma = ['https://www.football-lab.jp/gun/match/?year=2019', 'https://www.football-lab.jp/gun/match/?year=2020',
                          'https://www.football-lab.jp/gun/match/?year=2021', 'https://www.football-lab.jp/gun/match/']
        self.url_omiya = ['https://www.football-lab.jp/omiy/match/?year=2019', 'https://www.football-lab.jp/omiy/match/?year=2020',
                          'https://www.football-lab.jp/omiy/match/?year=2021', 'https://www.football-lab.jp/omiy/match/']
        self.url_chiba = ['https://www.football-lab.jp/chib/match/?year=2019', 'https://www.football-lab.jp/chib/match/?year=2020',
                          'https://www.football-lab.jp/chib/match/?year=2021', 'https://www.football-lab.jp/chib/match/']
        self.url_verdy = ['https://www.football-lab.jp/tk-v/match/?year=2019', 'https://www.football-lab.jp/tk-v/match/?year=2020',
                          'https://www.football-lab.jp/tk-v/match/?year=2021', 'https://www.football-lab.jp/tk-v/match/']
        self.url_machida = ['https://www.football-lab.jp/mcd/match/?year=2019', 'https://www.football-lab.jp/mcd/match/?year=2020',
                            'https://www.football-lab.jp/mcd/match/?year=2021', 'https://www.football-lab.jp/mcd/match/']
        self.url_yokohama = ['https://www.football-lab.jp/y-fc/match/?year=2019', 'https://www.football-lab.jp/y-fc/match/?year=2020',
                             'https://www.football-lab.jp/y-fc/match/?year=2021', 'https://www.football-lab.jp/y-fc/match/']
        self.url_kofu = ['https://www.football-lab.jp/kofu/match/?year=2019', 'https://www.football-lab.jp/kofu/match/?year=2020',
                         'https://www.football-lab.jp/kofu/match/?year=2021', 'https://www.football-lab.jp/kofu/match/']
        self.url_niigata = ['https://www.football-lab.jp/niig/match/?year=2019', 'https://www.football-lab.jp/niig/match/?year=2020',
                            'https://www.football-lab.jp/niig/match/?year=2021', 'https://www.football-lab.jp/niig/match/']
        self.url_kanazawa = ['https://www.football-lab.jp/kana/match/?year=2019', 'https://www.football-lab.jp/kana/match/?year=2020',
                             'https://www.football-lab.jp/kana/match/?year=2021', 'https://www.football-lab.jp/kana/match/']
        self.url_okayama = ['https://www.football-lab.jp/okay/match/?year=2019', 'https://www.football-lab.jp/okay/match/?year=2020',
                            'https://www.football-lab.jp/okay/match/?year=2021', 'https://www.football-lab.jp/okay/match/']
        self.url_yamaguchi = ['https://www.football-lab.jp/r-ya/match/?year=2019', 'https://www.football-lab.jp/r-ya/match/?year=2020',
                              'https://www.football-lab.jp/r-ya/match/?year=2021', 'https://www.football-lab.jp/r-ya/match/']
        self.url_tokushima = ['https://www.football-lab.jp/toku/match/?year=2019', 'https://www.football-lab.jp/toku/match/?year=2020',
                              'https://www.football-lab.jp/toku/match/?year=2021', 'https://www.football-lab.jp/toku/match/']
        self.url_nagasaki = ['https://www.football-lab.jp/ngsk/match/?year=2019', 'https://www.football-lab.jp/ngsk/match/?year=2020',
                             'https://www.football-lab.jp/ngsk/match/?year=2021', 'https://www.football-lab.jp/ngsk/match/']
        self.url_kumamoto = ['https://www.football-lab.jp/kuma/match/?year=2019', 'https://www.football-lab.jp/kuma/match/?year=2020',
                             'https://www.football-lab.jp/kuma/match/?year=2021', 'https://www.football-lab.jp/kuma/match/']
        self.url_oita = ['https://www.football-lab.jp/oita/match/?year=2019', 'https://www.football-lab.jp/oita/match/?year=2020',
                         'https://www.football-lab.jp/oita/match/?year=2021', 'https://www.football-lab.jp/oita/match/']
        self.url_agi = ['https://www.football-lab.jp/iwte/report/?year=2019&month=03&date=10', 'https://www.football-lab.jp/send/report/?year=2019&month=02&date=23',
                        'https://www.football-lab.jp/aki/report/?year=2019&month=03&date=10', 'https://www.football-lab.jp/yama/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/mito/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/toch/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/gun/report/?year=2019&month=03&date=10', 'https://www.football-lab.jp/omiy/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/chib/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/tk-v/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/mcd/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/y-fc/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/kofu/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/niig/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/kana/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/okay/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/r-ya/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/toku/report/?year=2019&month=02&date=24',
                        'https://www.football-lab.jp/ngsk/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/kuma/report/?year=2019&month=03&date=10',
                        'https://www.football-lab.jp/oita/report/?year=2019&month=02&date=23', 'https://www.football-lab.jp/ryuk/report/?year=2019&month=02&date=24']
        self.j2_league = [self.url_iwate, self.url_vegalta, self.url_akita, self.url_yamagata, self.url_mito, self.url_tochigi, self.url_gunma, self.url_omiya, self.url_chiba,
                          self.url_verdy, self.url_machida, self.url_yokohama, self.url_kofu, self.url_niigata, self.url_kanazawa, self.url_okayama, self.url_yamaguchi, self.url_tokushima,
                          self.url_nagasaki, self.url_kumamoto, self.url_oita, self.url_ryukyu]
        self.url_agi_ = ['https://www.football-lab.jp/y-fc/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/okay/report/?year=2019&month=02&date=24',
                         'https://www.football-lab.jp/toku/report/?year=2019&month=02&date=24', 'https://www.football-lab.jp/oita/report/?year=2019&month=02&date=23',
                         'https://www.football-lab.jp/ryuk/report/?year=2019&month=02&date=24']
        self.j2_teams = ['iwte', 'send', 'aki', 'yama', 'mito', 'toch', 'gun', 'omiy', 'chib', 'tk-v',
                         'mcd', 'y-fc', 'kofu', 'niig', 'kana', 'okay', 'r-ya', 'toku', 'ngsk', 'kuma', 'oita', 'ryuk']
        self.j2_names = ['いわてグルージャ盛岡', 'ベガルタ仙台', 'ブラウブリッツ秋田', 'モンテディオ山形', '水戸ホーリーホック', '栃木ＳＣ', 'ザスパクサツ群馬',
                         '大宮アルディージャ', 'ジェフユナイテッド千葉', '東京ヴェルディ', 'ＦＣ町田ゼルビア', '横浜ＦＣ', 'ヴァンフォーレ甲府', 'アルビレックス新潟',
                         'ツエーゲン金沢', 'ファジアーノ岡山', 'レノファ山口ＦＣ', '徳島ヴォルティス', 'Ｖ・ファーレン長崎', 'ロアッソ熊本', '大分トリニータ', 'ＦＣ琉球']

        self.j2_teams_ = ['y-fc', 'okay', 'toku', 'oita', 'ryuk']
        self.j2_names_ = ['横浜ＦＣ', 'ファジアーノ岡山', '徳島ヴォルティス', '大分トリニータ', 'ＦＣ琉球']
        self.where_league = [[1, 2, 2, 2], [1, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1],
                             [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1], [
                                 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1],
                             [1, 1, 1, 1], [1, 2, 2, 2], [1, 0, 0, 0], [1, 1, 1, 1]]
        self.where_league_ = [[1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]
        self.j1_games = [32, 38, 34, 34]
        self.j2_games = [42, 42, 42, 42]
        self.j3_games = [30, 28, 34, 34]
        self.rounds = []
        self.rounds_agi = []
        self.Hteam = []
        self.Hscore = []
        self.Ateam = []
        self.Ascore = []
        self.h_or_a = []
        self.WorL = []
        self.game_date = []
        self.wether = []
        self.grass = []
        self.spectators = []
        self.temperature = []
        self.tline1 = []
        self.tline2 = []
        self.tline3 = []
        self.tline4 = []
        self.tline5 = []
        self.tline6 = []
        self.Timeline = [self.tline1, self.tline2,
                         self.tline3, self.tline4, self.tline5, self.tline6]
        self.attack_cbp = []
        self.pass_cbp = []
        self.cross_cbp = []
        self.dribble_cbp = []
        self.shots_cbp = []
        self.score_cbp = []
        self.seizure = []
        self.diffence = []
        self.save_cbp = []
        self.Cbp = [self.attack_cbp, self.pass_cbp, self.cross_cbp, self.dribble_cbp,
                    self.shots_cbp, self.score_cbp, self.seizure, self.diffence, self.save_cbp]
        self.excepted_goal = []
        self.shots = []
        self.shots_success = []
        self.on_target = []
        self.pk = []
        self.passes = []
        self.pass_success = []
        self.cross = []
        self.cross_success = []
        self.d_fk = []
        self.i_fk = []
        self.ck = []
        self.throwin = []
        self.throwin_success = []
        self.dribble = []
        self.dribble_success = []
        self.tackle = []
        self.tackle_success = []
        self.clear = []
        self.intercept = []
        self.offside = []
        self.yellow = []
        self.red = []
        self.app_30 = []
        self.app_penalty = []
        self.attack_num = []
        self.chances = []
        self.control = []
        self.test = [4, 10, 12, 20, 22, 24]
        self.StatsH = [self.excepted_goal, self.shots_success, self.shots, self.on_target, self.pk, self.pass_success, self.passes, self.cross_success, self.cross, self.d_fk, self.i_fk, self.ck, self.throwin_success, self.throwin,
                       self.dribble_success, self.dribble, self.tackle_success, self.tackle, self.clear, self.intercept, self.offside, self.yellow, self.red, self.app_30, self.app_penalty, self.attack_num, self.chances, self.control]
        self.StatsA = [self.excepted_goal, self.shots, self.shots_success, self.on_target, self.pk, self.passes, self.pass_success, self.cross, self.cross_success, self.d_fk, self.i_fk, self.ck, self.throwin, self.throwin_success,
                       self.dribble, self.dribble_success, self.tackle, self.tackle_success, self.clear, self.intercept, self.offside, self.yellow, self.red, self.app_30, self.app_penalty, self.attack_num, self.chances, self.control]
        self.agi = []
        self.kagi = []
        self.agi_stats = [self.rounds_agi, self.agi, self.kagi]
        self.comment = [self.rounds, self.Hteam, self.Ateam, self.Hscore, self.Ascore, self.h_or_a, self.WorL, self.game_date, self.wether, self.grass, self.spectators, self.temperature, self.tline1, self.tline2, self.tline3, self.tline4, self.tline5, self.tline6, self.attack_cbp, self.pass_cbp, self.cross_cbp, self.dribble_cbp, self.shots_cbp,
                        self.score_cbp, self.seizure, self.diffence, self.save_cbp, self.excepted_goal, self.shots, self.shots_success, self.on_target, self.pk, self.passes, self.pass_success, self.cross, self.cross_success, self.d_fk, self.i_fk, self.ck, self.throwin, self.throwin_success, self.dribble, self.dribble_success, self.tackle, self.tackle_success, self.clear, self.intercept, self.offside, self.yellow, self.red, self.app_30, self.app_penalty, self.attack_num, self.chances,  self.control]

    def calc_games(self, t, j1):
        d = 0
        f = []
        for i in range(4):
            if self.where_league_[t][i] == 0:
                d += self.j1_games[i]
                f.insert(0, self.j1_games[i])
                j1.insert(0, 1)
            elif self.where_league_[t][i] == 1:
                d += self.j2_games[i]
                f.insert(0, self.j2_games[i])
                j1.insert(0, 0)
            else:
                d += self.j3_games[i]
                f.insert(0, self.j3_games[i])
                j1.insert(0, 0)
        return d, j1, f

    def get_agi(self, driver, n):
        x = 0
        # 第1節の試合結果画面
        get_stats = driver.find_element(
            By.CSS_SELECTOR, "#DataTables_Table_0 > tbody")
        for game_round in get_stats.find_elements(By.TAG_NAME, 'tr'):
            actions = ActionChains(driver)
            actions.move_to_element(game_round)
            actions.perform()
            for data in game_round.find_elements(By.TAG_NAME, 'td'):
                actions.move_to_element(data)
                actions.perform()
                if x == 0:
                    if len(data.text) > 0:
                        self.rounds_agi.append(data.text)
                elif x == 9:
                    if len(data.text) > 0:
                        self.agi.append(data.text)
                    else:
                        del self.rounds_agi[-1]
                elif x == 10:
                    self.kagi.append(data.text)
                elif x == 20:
                    x = -1
                x += 1
            n -= 1
            print(n)

    def get_header(self, driver, n, j1, gmnm, d):
        # 第1節の試合結果画面
        for _ in range(n):
            get_stats = driver.find_element(
                By.CSS_SELECTOR, f"#{self.j2_teams_[gmnm]} > article")
            t = get_stats.find_element(
                By.CSS_SELECTOR, f"#{self.j2_teams_[gmnm]} > article > div.vsHeader > table > tbody > tr:nth-child(3)")
            counter = 0
            for score in t.find_elements(By.TAG_NAME, "td"):
                actions = ActionChains(driver)
                actions.move_to_element(score)
                actions.perform()
                if counter == 0:            # home team
                    self.Hteam.append(score.text)
                    if self.Hteam[-1] == f"{self.j2_names_[gmnm]}":
                        self.h_or_a.append(1)
                elif counter == 1:          # home score
                    self.Hscore.append(score.text)
                elif counter == 3:          # away score
                    self.Ascore.append(score.text)
                elif counter == 4:          # away team
                    self.Ateam.append(score.text)
                    if self.Ateam[-1] == f"{self.j2_names_[gmnm]}":     # ホームアウェイ判定
                        self.h_or_a.append(0)
                        if int(self.Hscore[-1]) > int(self.Ascore[-1]):   # アウェイの場合の勝敗判定
                            self.WorL.append(0)
                        elif int(self.Hscore[-1]) < int(self.Ascore[-1]):
                            self.WorL.append(3)
                        else:
                            self.WorL.append(1)
                    else:                                       # ホームの場合の勝敗判定
                        if int(self.Hscore[-1]) < int(self.Ascore[-1]):
                            self.WorL.append(0)
                        elif int(self.Hscore[-1]) > int(self.Ascore[-1]):
                            self.WorL.append(3)
                        else:
                            self.WorL.append(1)
                counter += 1
            counter = 0

            t = get_stats.find_element(
                By.CSS_SELECTOR, f"#{self.j2_teams_[gmnm]} > article > div.boxHalfSP.l")
            self.game_date.append(t.text)       # 開催日程

            t = get_stats.find_element(
                By.CSS_SELECTOR, f"#{self.j2_teams_[gmnm]} > article > div.infoList")
            for env in t.find_elements(By.TAG_NAME, 'dl'):      # 環境情報
                actions = ActionChains(driver)
                actions.move_to_element(env)
                actions.perform()
                if counter == 0:
                    wthr = env.find_element(By.TAG_NAME, 'dd')
                    self.wether.append(wthr.text)
                elif counter == 1:
                    tmp = env.find_element(By.TAG_NAME, 'dd')
                    self.temperature.append(tmp.text)
                elif counter == 2:
                    grss = env.find_element(By.TAG_NAME, 'dd')
                    self.grass.append(grss.text)
                elif counter == 3:
                    spc = env.find_element(By.TAG_NAME, 'dd')
                    self.spectators.append(spc.text)
                counter += 1
            counter = -1

            # タイムラインの取得
            timeline = get_stats.find_element(
                By.CSS_SELECTOR, f"#{self.j2_teams_[gmnm]} > article > div:nth-child(10)")
            timel = []
            for tmline in timeline.find_elements(By.CLASS_NAME, 'boxTimeline'):
                for tl in tmline.find_elements(By.TAG_NAME, "tr"):    # 前半
                    actions = ActionChains(driver)
                    actions.move_to_element(tl)
                    actions.perform()
                    if counter % 4 == 0:
                        if self.h_or_a[-1] == 1:
                            poss = tl.find_element(
                                By.XPATH, 'td[1]')
                            timel.append(poss.text)
                        elif self.h_or_a[-1] == 0:
                            poss = tl.find_element(
                                By.XPATH, 'td[3]')
                            timel.append(poss.text)
                    counter += 1
                counter = -1

            counter = 1
            for i in range(6):          # tline1~6に代入
                self.Timeline[i].append(timel[i])

            # チャンスビルディングポイント
            game_cbp = []
            rnd = get_stats.find_element(
                By.XPATH, f'//*[@id="{self.j2_teams_[gmnm]}"]/article/div[10]/table[1]/thead/tr/th[3]')
            actions = ActionChains(driver)
            actions.move_to_element(rnd)
            actions.perform()
            self.rounds.append(rnd.text)
            t = get_stats.find_element(
                By.XPATH, '/html/body/article/div[10]/table[1]/tbody')
            for cbp in t.find_elements(By.TAG_NAME, 'tr'):
                if counter % 2 == 0:
                    x = 0
                    for data in cbp.find_elements(By.TAG_NAME, 'td'):
                        actions = ActionChains(driver)
                        actions.move_to_element(data)
                        actions.perform()
                        if self.h_or_a[-1] == 1:
                            if x == 2:
                                game_cbp.append(data.text)
                        elif self.h_or_a[-1] == 0:
                            if x == 4:
                                game_cbp.append(data.text)
                        x += 1
                counter += 1
            counter = 1
            for i in range(9):
                self.Cbp[i].append(game_cbp[i])

            # スタッツの取得
            x = 0
            game_stats = []
            t = get_stats.find_element(
                By.XPATH, '/html/body/article/div[10]/table[2]/tbody')
            for stats in t.find_elements(By.TAG_NAME, 'tr'):
                if counter % 2 == 0:
                    x = 0
                    for data in stats.find_elements(By.TAG_NAME, 'td'):
                        actions = ActionChains(driver)
                        actions.move_to_element(data)
                        actions.perform()
                        if self.h_or_a[-1] == 1:
                            if x == 2:
                                game_stats.append(data.text)
                            if counter in self.test:
                                if x == 1:
                                    game_stats.append(data.text)
                        elif self.h_or_a[-1] == 0:
                            if x == 4:
                                game_stats.append(data.text)
                            if counter in self.test:
                                if x == 5:
                                    game_stats.append(data.text)
                        x += 1
                counter += 1
            counter = 0

            if j1:      # リーグがj1の場合，スタッツに走行距離やスプリントのデータがあるため，取得しない
                for i in range(28):
                    if i >= 25:
                        if self.h_or_a[-1] == 1:
                            self.StatsH[i].append(game_stats[i+2])
                        elif self.h_or_a[-1] == 0:
                            self.StatsA[i].append(game_stats[i+2])
                    else:
                        if self.h_or_a[-1] == 1:
                            self.StatsH[i].append(game_stats[i])
                        elif self.h_or_a[-1] == 0:
                            self.StatsA[i].append(game_stats[i])
            else:
                for i in range(28):
                    if self.h_or_a[-1] == 1:
                        self.StatsH[i].append(game_stats[i])
                    elif self.h_or_a[-1] == 0:
                        self.StatsA[i].append(game_stats[i])
            time.sleep(0.3)

            # 次節に移動
            clk_next_game = driver.find_element(
                By.XPATH, "/html/body/article/div[6]/table/tbody/tr[1]/td/ul/li[3]")
            e = 0
            if clk_next_game.find_elements(By.TAG_NAME, 'a'):
                if _ == 6 and d == 3:
                    if e == 0:
                        driver.get(
                            "https://www.football-lab.jp/yama/report/?year=2022&month=04&date=10")
                        e += 1
                    elif e == 1:
                        driver.get(
                            "https://www.football-lab.jp/okay/report/?year=2022&month=04&date=09")
                else:
                    clk_next_game.click()
            else:
                break
            time.sleep(1)
            print(f'{n-_}, {j1}')


# %%
""" 
if __name__ == "__main__":
    time_start = time.time()
    for _ in range(22):
        getstts = GetStats()
        j1oroth = []
        n, j1_or_oth = getstts.calc_games(_, j1oroth)
        getstts.rounds = [r+1 for r in range(n)]
        options = Options()
        options.add_argument('--headless')        # ヘッドレス
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()                    # 画面最大化
        for i in range(4):
            driver.get(getstts.j2_league[_][i])
            getstts.get_agi(driver, n)
        with open(f'{getstts.j2_names_[_]}_stats_agi.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(getstts.header_agi)
            for i in range(len(getstts.agi_stats[0])):
                test = []
                for j in range(len(getstts.header_agi)):
                    test.append(getstts.agi_stats[j][i])
                writer.writerow(test)
        driver.get(f"{getstts.url_agi[_]}")
        time.sleep(1)
        for i in j1_or_oth:
            if i == 1:
                j1_flag = True
                getstts.get_header(driver, n, j1_flag, _)
            else:
                j1_flag = False
                getstts.get_header(driver, n, j1_flag, _)
        ''' for i in range(len(header)):
            comment[i].insert(0, header[i]) '''
        print(getstts.comment)
        with open(f'{getstts.j2_names_[_]}_stats_.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(getstts.header)
            for i in range(len(getstts.comment[0])):
                test = []
                for j in range(len(getstts.header)):
                    test.append(getstts.comment[j][i])
                writer.writerow(test)
        driver.quit()
    time_end = time.time()
    print(f"処理時間{time_end - time_start}")

# %%
# AGIの取得
for _ in range(22):
    getstts = GetStats()
    j1oroth = []
    n, j1_or_oth = getstts.calc_games(_, j1oroth)
    getstts.rounds = [r+1 for r in range(n)]
    options = Options()
    options.add_argument('--headless')        # ヘッドレス
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()                    # 画面最大化
    for i in range(4):
        driver.get(getstts.j2_league[_][i])
        getstts.get_agi(driver, n)
    with open(f'{getstts.j2_teams_[_]}_stats_agi.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(getstts.header_agi)
        for i in range(len(getstts.agi_stats[0])):
            test = []
            for j in range(len(getstts.header_agi)):
                test.append(getstts.agi_stats[j][i])
            writer.writerow(test)
 """
# %%
# statsの取得
for _ in range(5):
    time_start = time.time()
    getstts = GetStats()
    j1oroth = []
    d = 0
    n, j1_or_oth, gamenum = getstts.calc_games(_, j1oroth)
    print(n, j1_or_oth, gamenum)
    options = Options()
    options.add_argument('--headless')        # ヘッドレス
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()                    # 画面最大化
    driver.get(f"{getstts.url_agi_[_]}")
    time.sleep(0.2)
    for i in range(len(j1_or_oth)):
        if j1_or_oth[i] == 1:
            j1_flag = True
            getstts.get_header(driver, gamenum[i], j1_flag, _, d)
            if _ == 1:
                d += 1
        else:
            j1_flag = False
            getstts.get_header(driver, gamenum[i], j1_flag, _, d)
            if _ == 1:
                d += 1
    with open(f'{getstts.j2_teams_[_]}_stats_.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(getstts.header)
        for i in range(n):
            test = []
            for j in range(len(getstts.header)):
                test.append(getstts.comment[j][i])
            writer.writerow(test)
    driver.quit()
    time_end = time.time()
    print(f"処理時間{time_end - time_start}")

# %%
# %%

# %%
# %%
""" time_start = time.time()
getstts = GetStats()
j1oroth = []
_ = 3
d = 0
n, j1_or_oth, gamenum = getstts.calc_games(_, j1oroth)
print(n, j1_or_oth, gamenum)
options = Options()
options.add_argument('--headless')        # ヘッドレス
driver = webdriver.Chrome(options=options)
driver.maximize_window()                    # 画面最大化
driver.get(f"{getstts.url_agi[_]}")
time.sleep(0.2)
for i in range(len(j1_or_oth)):
    if j1_or_oth[i] == 1:
        j1_flag = True
        getstts.get_header(driver, gamenum[i], j1_flag, _, d)
        d += 1
    else:
        j1_flag = False
        getstts.get_header(driver, gamenum[i], j1_flag, _, d)
        d += 1
with open(f'{getstts.j2_teams_[_]}_stats_.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(getstts.header)
    for i in range(n):
        test = []
        for j in range(len(getstts.header)):
            test.append(getstts.comment[j][i])
        writer.writerow(test)
driver.quit()
time_end = time.time()
print(f"処理時間{time_end - time_start}")
 """
# %%
