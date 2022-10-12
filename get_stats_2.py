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


header = ['Round', 'Hteam', 'Ateam', 'Hscore', 'Ascore', 'HorA', 'WorL', 'Date', 'Wether', 'Grass_condition', 'Spectators', 'temperature', 'TimeLine1', 'TimeLine2', 'TimeLine3',
          'TimeLine4', 'TimeLine5', 'TimeLine6', 'Attack_CBP', 'Pass_CBP', 'Cross_CBP', 'Dribble_CBP', 'Shots_CBP', 'Score_CBP', 'Seizure', 'Diffence', 'Save_CBP', 'Expect_Goal',
          'Shots', 'Shots_Success', 'On_Target', 'PK', 'Pass', 'Pass_Success', 'Cross', 'Cross_Success', 'D_FK', 'I_FK', 'CK', 'Throwing', 'Throwing_Success', 'Dribble', 'Dribble_Success', 'Tackle', 'Tackle_Success', 'Clear', 'Intercept', 'OffSide', 'Yellow', 'Red', 'Approach_30m', 'Approach_Penalty', 'Attack_Num', 'Chances', 'Control', 'AGI', 'KAGI']  # 'AGI', 'KAGI'の追加も

url = ['https://www.football-lab.jp/ryuk/match/?year=2019', 'https://www.football-lab.jp/ryuk/match/?year=2020',
       'https://www.football-lab.jp/ryuk/match/?year=2021', 'https://www.football-lab.jp/ryuk/match/']
rounds = [i+1 for i in range(165)]
Hteam = []
Hscore = []
Ateam = []
Ascore = []
h_or_a = []
WorL = []
game_date = []
wether = []
grass = []
spectators = []
temperature = []
tline1 = []
tline2 = []
tline3 = []
tline4 = []
tline5 = []
tline6 = []
Timeline = [tline1, tline2, tline3, tline4, tline5, tline6]
attack_cbp = []
pass_cbp = []
cross_cbp = []
dribble_cbp = []
shots_cbp = []
score_cbp = []
seizure = []
diffence = []
save_cbp = []
Cbp = [attack_cbp, pass_cbp, cross_cbp, dribble_cbp,
       shots_cbp, score_cbp, seizure, diffence, save_cbp]
excepted_goal = []
shots = []
shots_success = []
on_target = []
pk = []
passes = []
pass_success = []
cross = []
cross_success = []
d_fk = []
i_fk = []
ck = []
throwin = []
throwin_success = []
dribble = []
dribble_success = []
tackle = []
tackle_success = []
clear = []
intercept = []
offside = []
yellow = []
red = []
app_30 = []
app_penalty = []
attack_num = []
chances = []
control = []
test = [4, 10, 12, 20, 22, 24]
StatsH = [excepted_goal, shots_success, shots, on_target, pk, pass_success, passes, cross_success, cross, d_fk, i_fk, ck, throwin_success, throwin,
          dribble_success, dribble, tackle_success, tackle, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances, control]
StatsA = [excepted_goal, shots, shots_success, on_target, pk, passes, pass_success, cross, cross_success, d_fk, i_fk, ck, throwin, throwin_success,
          dribble, dribble_success, tackle, tackle_success, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances, control]
agi = []
kagi = []
comment = [rounds, Hteam, Ateam, Hscore, Ascore, h_or_a, WorL, game_date, wether, grass, spectators, temperature, tline1, tline2, tline3, tline4, tline5, tline6, attack_cbp, pass_cbp, cross_cbp, dribble_cbp, shots_cbp,
           score_cbp, seizure, diffence, save_cbp, excepted_goal, shots, shots_success, on_target, pk, passes, pass_success, cross, cross_success, d_fk, i_fk, ck, throwin, throwin_success, dribble, dribble_success, tackle, tackle_success, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances,  control, agi, kagi]


def get_agi(driver, n):
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
            if x == 9:
                agi.append(data.text)
            elif x == 10:
                kagi.append(data.text)
            elif x == 20:
                x = -1
            x += 1


def get_header(driver, n):
    # 第1節の試合結果画面
    for _ in range(n):
        get_stats = driver.find_element(
            By.CSS_SELECTOR, "#ryuk > article")
        actions = ActionChains(driver)
        actions.move_to_element(get_stats)
        actions.perform()
        t = get_stats.find_element(
            By.CSS_SELECTOR, "#ryuk > article > div.vsHeader > table > tbody > tr:nth-child(3)")
        counter = 0
        for score in t.find_elements(By.TAG_NAME, "td"):
            if counter == 0:            # home team
                Hteam.append(score.text)
                if Hteam[-1] == "ＦＣ琉球":
                    h_or_a.append(1)
            elif counter == 1:          # home score
                Hscore.append(score.text)
            elif counter == 3:          # away score
                Ascore.append(score.text)
            elif counter == 4:          # away team
                Ateam.append(score.text)
                if Ateam[-1] == "ＦＣ琉球":     # ホームアウェイ判定
                    h_or_a.append(0)
                    if int(Hscore[-1]) > int(Ascore[-1]):   # アウェイの場合の勝敗判定
                        WorL.append(0)
                    elif int(Hscore[-1]) < int(Ascore[-1]):
                        WorL.append(3)
                    else:
                        WorL.append(0)
                else:                                       # ホームの場合の勝敗判定
                    if int(Hscore[-1]) < int(Ascore[-1]):
                        WorL.append(0)
                    elif int(Hscore[-1]) > int(Ascore[-1]):
                        WorL.append(3)
                    else:
                        WorL.append(0)
            counter += 1
        counter = 0
        time.sleep(1)

        t = get_stats.find_element(By.CSS_SELECTOR, "#ryuk > article > div.boxHalfSP.l")
        game_date.append(t.text)       # 開催日程

        t = get_stats.find_element(By.CSS_SELECTOR, "#ryuk > article > div.infoList")
        for env in t.find_elements(By.TAG_NAME, 'dl'):      # 環境情報
            if counter == 0:
                wthr = env.find_element(By.TAG_NAME, 'dd')
                wether.append(wthr.text)
            elif counter == 1:
                tmp = env.find_element(By.TAG_NAME, 'dd')
                temperature.append(tmp.text)
            elif counter == 2:
                grss = env.find_element(By.TAG_NAME, 'dd')
                grass.append(grss.text)
            elif counter == 3:
                spc = env.find_element(By.TAG_NAME, 'dd')
                spectators.append(spc.text)
            counter += 1
        counter = -1
        time.sleep(1)
# ryuk > article > div:nth-child(10) > div:nth-child(16) > table > tbody
        # タイムラインの取得
        timeline = get_stats.find_element(
            By.CSS_SELECTOR, "#ryuk > article > div:nth-child(10)")
        timel = []
        for tmline in timeline.find_elements(By.CLASS_NAME, 'boxTimeline'):
            for tl in tmline.find_elements(By.TAG_NAME, "tr"):    # 前半
                if counter % 4 == 0:
                    if h_or_a[-1] == 1:
                        poss = tl.find_element(
                            By.XPATH, 'td[1]')
                        timel.append(poss.text)
                    elif h_or_a[-1] == 0:
                        poss = tl.find_element(
                            By.XPATH, 'td[3]')
                        timel.append(poss.text)
                counter += 1
            counter = -1

        counter = 1
        for i in range(6):          # tline1~6に代入
            Timeline[i].append(timel[i])
        time.sleep(1)

        # チャンスビルディングポイント
        game_cbp = []
        t = get_stats.find_element(
            By.XPATH, '/html/body/article/div[10]/table[1]/tbody')
        for cbp in t.find_elements(By.TAG_NAME, 'tr'):
            if counter % 2 == 0:
                x = 0
                for data in cbp.find_elements(By.TAG_NAME, 'td'):
                    if h_or_a[-1] == 1:
                        if x == 2:
                            game_cbp.append(data.text)
                    elif h_or_a[-1] == 0:
                        if x == 4:
                            game_cbp.append(data.text)
                    x += 1
            counter += 1
        counter = 1
        for i in range(9):
            Cbp[i].append(game_cbp[i])
        time.sleep(1)

        # スタッツの取得
        x = 0
        game_stats = []
        t = get_stats.find_element(
            By.XPATH, '/html/body/article/div[10]/table[2]/tbody')
        for stats in t.find_elements(By.TAG_NAME, 'tr'):
            if counter % 2 == 0:
                x = 0
                for data in stats.find_elements(By.TAG_NAME, 'td'):
                    if h_or_a[-1] == 1:
                        if x == 2:
                            game_stats.append(data.text)
                        if counter in test:
                            if x == 1:
                                game_stats.append(data.text)
                    elif h_or_a[-1] == 0:
                        if x == 4:
                            game_stats.append(data.text)
                        if counter in test:
                            if x == 5:
                                game_stats.append(data.text)
                    x += 1
            counter += 1
        counter = 0

        for i in range(28):
            if h_or_a[-1] == 1:
                StatsH[i].append(game_stats[i])
            elif h_or_a[-1] == 0:
                StatsA[i].append(game_stats[i])
        time.sleep(1)

        # 次節に移動
        clk_next_game = driver.find_element(
            By.XPATH, "/html/body/article/div[6]/table/tbody/tr[1]/td/ul/li[3]")
        if clk_next_game.find_elements(By.TAG_NAME, 'a'):
            clk_next_game.click()
        else:
            break
        time.sleep(1)
        print(_)


if __name__ == "__main__":
    n = 165
    time_start = time.time()
    options = Options()
    options.add_argument('--headless')        # ヘッドレス
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()                    # 画面最大化
    for i in range(4):
        driver.get(url[i])
        get_agi(driver, n)
    driver.get("https://www.football-lab.jp/ryuk/report/?year=2019&month=02&date=24")
    time.sleep(1)
    get_header(driver, n)
    for i in range(len(header)):
        comment[i].insert(0, header[i])
    with open('fcryukyu_stats_.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(comment)
    time_end = time.time()
    print(f"処理時間{time_end - time_start}")
    driver.quit()
