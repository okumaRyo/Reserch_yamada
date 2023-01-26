import csv
import time
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

"""
    チーム変更の際に変更する値
        links                   urlのチーム名のみの変更で可
        where_league            2019, 2020, 2021, 2022の順番。get_stats4.pyの逆順
        if home[-1] == "**":    ** -> チーム名(琉球, 大分, etc)
        with open(****          出力するファイル名(J2_commentフォルダの中に作成)

        *//J3は監督コメント無し
    """
w = []
header = ['Round', 'Hteam', 'Ateam', 'Hscore', 'Ascore', 'WorL',
          'Preview', 'Review', 'H_comment', 'A_comment']
g_round = []
home = []
away = []
home_score = []
away_score = []
midokoro = []
review = []
h_comment = []
a_comment = []
winlose = []
sokuhou = []
comment = [g_round, home, away, home_score, away_score,
           winlose, midokoro, review, h_comment, a_comment]
links = ['https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=morioka&year=2019',
         'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=morioka&year=2020',
         'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=morioka&year=2021',
         'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=morioka&year=2022']

links_ = ['https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=sendai&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=sendai&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=sendai&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=sendai&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=akita&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=akita&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=akita&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=akita&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamagata&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamagata&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamagata&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamagata&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=mito&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=mito&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=mito&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=mito&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tochigi&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tochigi&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tochigi&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tochigi&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kusatsu&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kusatsu&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kusatsu&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kusatsu&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=omiya&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=omiya&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=omiya&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=omiya&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=chiba&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=chiba&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=chiba&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=chiba&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokyov&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokyov&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokyov&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokyov&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=machida&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=machida&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=machida&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=machida&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yokohamafc&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yokohamafc&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yokohamafc&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yokohamafc&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kofu&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kofu&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kofu&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kofu&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=niigata&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=niigata&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=niigata&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=niigata&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kanazawa&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kanazawa&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kanazawa&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kanazawa&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=okayama&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=okayama&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=okayama&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=okayama&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamaguchi&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamaguchi&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamaguchi&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=yamaguchi&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokushima&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokushima&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokushima&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=tokushima&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=nagasaki&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=nagasaki&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=nagasaki&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=nagasaki&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kumamoto&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kumamoto&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kumamoto&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=kumamoto&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=oita&year=2019',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=oita&year=2020',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=oita&year=2021',
          'https://www.jleague.jp/match/search/?category%5B%5D=j1&category%5B%5D=j2&category%5B%5D=j3&club%5B%5D=oita&year=2022',
          'https://www.jleague.jp/match/search/?category%5B%5D=j2&club%5B%5D=ryukyu&year=2019&section=',
          'https://www.jleague.jp/match/search/?category%5B%5D=j2&club%5B%5D=ryukyu&year=2020&section=',
          'https://www.jleague.jp/match/search/?category%5B%5D=j2&club%5B%5D=ryukyu&year=2021&section=',
          'https://www.jleague.jp/match/search/?category%5B%5D=j2&club%5B%5D=ryukyu&year=2022&section=']

where_league = [2, 2, 2, 1]
j1_games = [34, 34, 38, 34]
j2_games = [42, 42, 42, 42]
j3_games = [34, 34, 28, 34]


def set_window(driver, n):
    move_first_game = driver.find_element(By.ID, "day_month")
    sel_move_first_game = Select(move_first_game)
    sel_move_first_game.select_by_value(str(n))
    time.sleep(1.5)
    clk_first_game = driver.find_element(By.ID, "day_s_btn")
    clk_first_game.click()
    time.sleep(1.5)


def get_comment(driver, n):
    x = 0
    s = ""
    game_num = 0
    switch_num = 0
    # 第1節の試合の画面にする

    # 第1節の試合結果画面
    game_rounds = driver.find_element(
        By.CSS_SELECTOR, "body > div.content > div.main > section > section")
    a = game_rounds.find_elements(By.TAG_NAME, 'section')
    b = [i for i in a]
    for game_round in b:

        if game_num == switch_num:
            games = game_round.find_element(By.CLASS_NAME, 'match')
            H_team = game_round.find_element(
                By.CSS_SELECTOR, 'table > tbody > tr > td.match > a > table > tbody > tr > td.clubName.leftside')
            home.append(H_team.text)
            A_team = game_round.find_element(
                By.CSS_SELECTOR, 'table > tbody > tr > td.match > a > table > tbody > tr > td.clubName.rightside')
            away.append(A_team.text)
            url = games.find_element(By.TAG_NAME, 'a')
            r = url.get_attribute('href')
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(str(r))
            time.sleep(2.0)

            g_Round = driver.find_element(By.CLASS_NAME, "matchVsTitle__league")
            g1_H_score = driver.find_element(By.CLASS_NAME, "leagLeftScore")
            # print(f"Home_Score is {g1_H_score.text}")
            g1_A_score = driver.find_element(By.CLASS_NAME, "leagRightScore")
            # print(f"Away_Score is {g1_A_score.text}")
            g_round.append(g_Round.text)
            home_score.append(g1_H_score.text)
            away_score.append(g1_A_score.text)

            if home[-1] == "岩手":
                if home_score[-1] > away_score[-1]:
                    winlose.append(3)
                elif home_score[-1] == away_score[-1]:
                    winlose.append(1)
                else:
                    winlose.append(0)
            else:
                if home_score[-1] < away_score[-1]:
                    winlose.append(3)
                elif home_score[-1] == away_score[-1]:
                    winlose.append(1)
                else:
                    winlose.append(0)
            # 試合速報(未完成。データからの学習ができるようになった後に作成？)
            """ driver.find_element(
                By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav:nth-child(7) > ul > li:nth-child(3)').click()
            lives = driver.find_element(By.CSS_SELECTOR, '#loadArea > section')
            for live in lives.find_elements(By.TAG_NAME, 'ul'):
                spot = live.find_element(By.CSS_SELECTOR, 'div.spotRightTxt')
                sokuhou.append(spot.text)
            print(sokuhou)
            time.sleep(2.0) """
            # 見どころのテキストを取得
            # print(winlose[-1])
            driver.find_element(By.CSS_SELECTOR,        # 見どころの表示クリック
                                "body > div.content.clearfix > div.main > section > nav.tabNavArea.matchTimeLineNav > ul > li:nth-child(1)").click()
            time.sleep(2.0)
            g1_midokoro = WebDriverWait(driver, 10).until(lambda x: x.find_element(
                By.XPATH, '//*[@id="loadArea"]/section[1]/div[2]'))
            g1_midokoro = g1_midokoro.text

            # print(f'[見どころ] {g1_midokoro.text}')
            time.sleep(2.0)
            # 試合後のテキストを取得
            driver.find_element(
                By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav.tabNavArea.matchTimeLineNav > ul > li:nth-child(3)').click()
            time.sleep(2.0)
            g1_report = WebDriverWait(driver, 10).until(lambda x: x.find_element(       # レビューのテキストを取得
                By.CSS_SELECTOR, '#loadArea > section > div.warDataTxt'))
            g1_report = g1_report.text
            # print(f'[レビュー] {g1_report.text}')
            time.sleep(2.0)
            # 監督コメントの取得
            cmnt = driver.find_element(
                By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav:nth-child(8) > ul > li:nth-child(2)')
            link_enable = cmnt.get_attribute('class')
            if link_enable == 'noLink':
                g1_H_cmnt = 'コメントなし'
                g1_A_cmnt = 'コメントなし'
            else:
                driver.find_element(
                    By.CSS_SELECTOR, 'body > div.content.clearfix > div.main > section > nav:nth-child(8) > ul > li:nth-child(2)').click()
                time.sleep(2.0)
                g1_H_cmnt = WebDriverWait(driver, 10).until(                                # ホーム監督のコメント
                    lambda x: x.find_element(By.XPATH, '//*[@id="loadArea"]/div/div[1]/p'))
                g1_H_cmnt = g1_H_cmnt.text
                g1_A_cmnt = WebDriverWait(driver, 10).until(                                # アウェイ監督のコメント
                    lambda x: x.find_element(By.XPATH, '//*[@id="loadArea"]/div/div[2]/p'))
                g1_A_cmnt = g1_A_cmnt.text
            # print(f'[Home監督コメント] {g1_H_cmnt}\n[Away監督コメント] {g1_A_cmnt}')
            midokoro.append(g1_midokoro.strip())
            review.append(g1_report.strip())
            h_comment.append(g1_H_cmnt.strip())
            a_comment.append(g1_A_cmnt.strip())
            game_num += 1
            if where_league[n] == 0:
                if game_num >= j1_games[n]:
                    switch_num = 50
            elif where_league[n] == 1:
                if game_num >= j2_games[n]:
                    switch_num = 50
            elif where_league[n] == 2:
                if game_num >= j3_games[n]:
                    switch_num = 50
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            # time.sleep(2.0)
            # print(f'{x}{H_team}vs{A_team}{game_round}')

            time.sleep(2.0)
        elif game_num < switch_num:
            break
        switch_num += 1
        print(switch_num, g_round[-1])

    """ for i in midokoro:
        print(f'{i}\n-----------------------------')
        f.write(f'__{i}__\n') """

    # print(f'<Home>\n{home}\n<Away>\n{away}\n<見どころ>\n{midokoro}\n<レビュー>\n{review}\n<ホームコメント>\n{h_comment}\n<アウェイコメント>\n{a_comment}')
    # print(midokoro)


if __name__ == "__main__":
    time_start = time.time()
    for _ in range(4):
        options = Options()
        options.add_argument('--headless')        # ヘッドレス
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()                    # 画面最大化
        driver.get(f"{links[_]}")
        get_comment(driver, _)
        with open('J2_comment/morioka_comment_.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(home)):
                test = []
                for j in range(len(header)):
                    test.append(comment[j][i])
                writer.writerow(test)
        driver.quit()
    time_end = time.time()
    print(f"処理時間{time_end - time_start}")
