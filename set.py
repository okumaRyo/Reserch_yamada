import csv
import time
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options

home = []
away = []
home_score = []
away_score = []
midokoro = []
review = []
h_coment = []
a_coment = []
coment = [home, away, home_score, away_score, midokoro, review, h_coment, a_coment]


def get_coment(driver, n):
    x = 0
    s = 0
    game_num = 0
    switch_num = 0
    # 第1節の試合の画面にする
    move_first_game = driver.find_element(By.ID, "day_month")
    sel_move_first_game = Select(move_first_game)
    sel_move_first_game.select_by_value(str(n))
    time.sleep(1)
    clk_first_game = driver.find_element(By.ID, "day_s_btn")
    clk_first_game.click()
    time.sleep(1)
    # ホーム・アウェイチーム名の取得
    """ g1_H_tm = driver.find_element(By.CSS_SELECTOR,
                                  "#loadArea > section > section:nth-child(4) > div > ul > li.vsTxt > div > span:nth-child(1)")
    # print(f"G1_H_team is {g1_H_tm.text}")
    g1_A_tm = driver.find_element(By.CSS_SELECTOR,
                                  "#loadArea > section > section:nth-child(4) > div > ul > li.vsTxt > div > span:nth-child(3)")
    # print(f"G1_A_team is {g1_A_tm.text}")
    elmnts = driver.find_element(By.CSS_SELECTOR, '#loadArea > section')
    for elm in elmnts.find_elements(By.TAG_NAME, 'section'):
        tm = elm.find_element(By.CLASS_NAME, 'vsTxt')
        for team in tm.find_elements(By.TAG_NAME, 'span'):
            if s == 0:
                H_team = team.text
            if s == 2:
                A_team = team.text
            s += 1
        print(f'{x}{H_team}vs{A_team}')
        x += 1
        s = 0
    time.sleep(1) """

    # 第1節の試合結果画面
    game_rounds = driver.find_element(By.CSS_SELECTOR, "#loadArea > section")
    for game_round in game_rounds.find_elements(By.TAG_NAME, 'section'):

        if game_num == switch_num:
            games = game_round.find_element(By.CLASS_NAME, 'vsTxt')
            for game in games.find_elements(By.TAG_NAME, 'span'):
                if s == 0:
                    H_team = game.text
                    home.append(H_team)
                if s == 2:
                    A_team = game.text
                    away.append(A_team)
                if s == 1:
                    url = game.find_element(By.TAG_NAME, 'a')
                    r = url.get_attribute('href')
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[1])
                    driver.get(str(r))
                    time.sleep(1)
                    g1_H_score = driver.find_element(By.CLASS_NAME, "leagLeftScore")
                    # print(f"Home_Score is {g1_H_score.text}")
                    g1_A_score = driver.find_element(By.CLASS_NAME, "leagRightScore")
                    # print(f"Away_Score is {g1_A_score.text}")
                    home_score.append(g1_H_score.text)
                    away_score.append(g1_A_score.text)

                    # 見どころのテキストを取得
                    driver.find_element(By.CSS_SELECTOR,        # 見どころの表示クリック
                                        "body > div.content.clearfix > div.main > section > nav.tabNavArea.matchTimeLineNav > ul > li:nth-child(1)").click()
                    time.sleep(1)
                    g1_midokoro = WebDriverWait(driver, 10).until(lambda x: x.find_element(
                        By.XPATH, '//*[@id="loadArea"]/section[1]/div[2]'))
                    g1_midokoro = g1_midokoro.text

                    # print(f'[見どころ] {g1_midokoro.text}')
                    time.sleep(1)
                    # 試合後のテキストを取得
                    driver.find_element(
                        By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav.tabNavArea.matchTimeLineNav > ul > li:nth-child(3)').click()
                    time.sleep(1)
                    g1_report = WebDriverWait(driver, 10).until(lambda x: x.find_element(       # レビューのテキストを取得
                        By.XPATH, '//*[@id="loadArea"]/section/div[2]'))
                    g1_report = g1_report.text
                    # print(f'[レビュー] {g1_report.text}')
                    time.sleep(1)
                    # 監督コメントの取得
                    driver.find_element(
                        By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav:nth-child(8) > ul > li:nth-child(2)').click()
                    time.sleep(1)
                    g1_H_cmnt = WebDriverWait(driver, 10).until(                                # ホーム監督のコメント
                        lambda x: x.find_element(By.XPATH, '//*[@id="loadArea"]/div/div[1]/p'))
                    g1_H_cmnt = g1_H_cmnt.text
                    g1_A_cmnt = WebDriverWait(driver, 10).until(                                # アウェイ監督のコメント
                        lambda x: x.find_element(By.XPATH, '//*[@id="loadArea"]/div/div[2]/p'))
                    g1_A_cmnt = g1_A_cmnt.text
                    # print(f'[Home監督コメント] {g1_H_cmnt.text}\n[Away監督コメント] {g1_A_cmnt.text}')
                    midokoro.append(g1_midokoro.strip())
                    review.append(g1_report.strip())
                    h_coment.append(g1_H_cmnt.strip())
                    a_coment.append(g1_A_cmnt.strip())
                    game_num += 1
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)

                s += 1
            s = 0
            # print(f'{x}{H_team}vs{A_team}{game_round}')

            time.sleep(1)
        switch_num += 1
        print(coment)
    """ for i in midokoro:
        print(f'{i}\n-----------------------------')
        f.write(f'__{i}__\n') """
    # print(f'<Home>\n{home}\n<Away>\n{away}\n<見どころ>\n{midokoro}\n<レビュー>\n{review}\n<ホームコメント>\n{h_coment}\n<アウェイコメント>\n{a_coment}')
    # print(midokoro)


if __name__ == "__main__":
    n = 2
    options = Options()
    # options.add_argument('--headless')        # ヘッドレス
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()                    # 画面最大化
    driver.get("https://www.jleague.jp/club/ryukyu/day/#day")
    time.sleep(1)
    while n < 9:
        get_coment(driver, n)
        n += 1
    with open('fcryukyu2021.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(coment)
    driver.quit()
