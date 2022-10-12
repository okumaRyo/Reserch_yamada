import csv
import time
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


header = ['round', 'date', 'day', 'team', 'score', 'HorA', 'stadium', 'spectator', 'wether', 'AGI', 'KAGI', 'chances',
          'shots', 'shot_success', 'control', 'attack_cbp', 'pass_cbp', 'seizure', 'diffence', 'scorers', 'bosses']
rounds = []
game_dates = []
game_days = []
opponents = []
game_scores = []
h_or_a = []
stadiums = []
spectators = []
wether = []
agi = []
kagi = []
chances = []
shots = []
shots_successes = []
control = []
attack_cbp = []
pass_cbp = []
seizure = []
diffence = []
scorers = []
bosses = []
comment = [rounds, game_dates, game_days, opponents, game_scores, h_or_a, stadiums, spectators, wether,
           agi, kagi, chances, shots, shots_successes, control, attack_cbp, pass_cbp, seizure, diffence, scorers, bosses]


def set_window(driver, n):
    # 第1節の試合結果画面
    game_header = driver.find_element(
        By.CSS_SELECTOR, "#DataTables_Table_0_wrapper > div > div.dataTables_scroll > div.dataTables_scrollHead > div > table > thead > tr")
    for game_round in game_header.find_elements(By.TAG_NAME, 'th'):
        actions = ActionChains(driver)
        actions.move_to_element(game_round)
        actions.perform()
        header.append(game_round.text)
        print(header)


def get_header(driver, n):
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
                rounds.append(data.text)
            elif x == 1:
                game_dates.append(data.text)
            elif x == 2:
                game_days.append(data.text)
            elif x == 3:
                opponents.append(data.text)
            elif x == 4:
                game_scores.append(data.text)
            elif x == 5:
                h_or_a.append(data.text)
            elif x == 6:
                stadiums.append(data.text)
            elif x == 7:
                spectators.append(data.text)
            elif x == 8:
                wether.append(data.text)
            elif x == 9:
                agi.append(data.text)
            elif x == 10:
                kagi.append(data.text)
            elif x == 11:
                chances.append(data.text)
            elif x == 12:
                shots.append(data.text)
            elif x == 13:
                shots_successes.append(data.text)
            elif x == 14:
                control.append(data.text)
            elif x == 15:
                attack_cbp.append(data.text)
            elif x == 16:
                pass_cbp.append(data.text)
            elif x == 17:
                seizure.append(data.text)
            elif x == 18:
                diffence.append(data.text)
            elif x == 19:
                scorers.append(data.text)
            elif x == 20:
                bosses.append(data.text)
                x = -1
            x += 1
# 不要な情報はとってこない!!! ex 日付やスタジアムなど


# day_s_btn
# loadArea > section > section:nth-child(4) > div > ul > li.vsTxt > div > span.score > a
# loadArea > section > section:nth-child(5) > div > ul > li.vsTxt > div > span:nth-child(1)
# loadArea > section > section:nth-child(4) > div > ul > li.vsTxt > div > span:nth-child(1)
# tabpanelTopics1
if __name__ == "__main__":
    n = 2
    time_start = time.time()
    options = Options()
    # options.add_argument('--headless')        # ヘッドレス
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()                    # 画面最大化
    driver.get("https://www.football-lab.jp/ryuk/match/")
    time.sleep(1)
    get_header(driver, n)
    for i in range(len(header)):
        comment[i].insert(0, header[i])
    with open('fcryukyu_stats.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(comment)
    time_end = time.time()
    print(f"処理時間{time_end - time_start}")
    driver.quit()
