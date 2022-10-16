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
Hteam = [i+10 for i in range(165)]
Hscore = [i+100 for i in range(165)]
Ateam = [i+1000 for i in range(165)]
Ascore = [i+12 for i in range(165)]
h_or_a = [i+13 for i in range(165)]
WorL = [i+14 for i in range(165)]
game_date = [i+1 for i in range(165)]
wether = [i+1 for i in range(165)]
grass = [i+1 for i in range(165)]
spectators = [i+1 for i in range(165)]
temperature = [i+1 for i in range(165)]
tline1 = [i+1 for i in range(165)]
tline2 = [i+1 for i in range(165)]
tline3 = [i+1 for i in range(165)]
tline4 = [i+1 for i in range(165)]
tline5 = [i+1 for i in range(165)]
tline6 = [i+1 for i in range(165)]
Timeline = [tline1, tline2, tline3, tline4, tline5, tline6]
attack_cbp = [i+1 for i in range(165)]
pass_cbp = [i+1 for i in range(165)]
cross_cbp = [i+1 for i in range(165)]
dribble_cbp = [i+1 for i in range(165)]
shots_cbp = [i+1 for i in range(165)]
score_cbp = [i+1 for i in range(165)]
seizure = [i+1 for i in range(165)]
diffence = [i+1 for i in range(165)]
save_cbp = [i+1 for i in range(165)]
Cbp = [attack_cbp, pass_cbp, cross_cbp, dribble_cbp,
       shots_cbp, score_cbp, seizure, diffence, save_cbp]
excepted_goal = [i+1 for i in range(165)]
shots = [i+1 for i in range(165)]
shots_success = [i+1 for i in range(165)]
on_target = [i+1 for i in range(165)]
pk = [i+1 for i in range(165)]
passes = [i+1 for i in range(165)]
pass_success = [i+1 for i in range(165)]
cross = [i+1 for i in range(165)]
cross_success = [i+1 for i in range(165)]
d_fk = [i+1 for i in range(165)]
i_fk = [i+1 for i in range(165)]
ck = [i+1 for i in range(165)]
throwin = [i+1 for i in range(165)]
throwin_success = [i+1 for i in range(165)]
dribble = [i+1 for i in range(165)]
dribble_success = [i+1 for i in range(165)]
tackle = [i+1 for i in range(165)]
tackle_success = [i+1 for i in range(165)]
clear = [i+1 for i in range(165)]
intercept = [i+1 for i in range(165)]
offside = [i+1 for i in range(165)]
yellow = [i+1 for i in range(165)]
red = [i+1 for i in range(165)]
app_30 = [i+1 for i in range(165)]
app_penalty = [i+1 for i in range(165)]
attack_num = [i+1 for i in range(165)]
chances = [i+1 for i in range(165)]
control = [i+1 for i in range(165)]
test = [4, 10, 12, 20, 22, 24]
StatsH = [excepted_goal, shots_success, shots, on_target, pk, pass_success, passes, cross_success, cross, d_fk, i_fk, ck, throwin_success, throwin,
          dribble_success, dribble, tackle_success, tackle, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances, control]
StatsA = [excepted_goal, shots, shots_success, on_target, pk, passes, pass_success, cross, cross_success, d_fk, i_fk, ck, throwin, throwin_success,
          dribble, dribble_success, tackle, tackle_success, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances, control]
agi = [i+1 for i in range(165)]
kagi = [i+1 for i in range(165)]
comment = [rounds, Hteam, Ateam, Hscore, Ascore, h_or_a, WorL, game_date, wether, grass, spectators, temperature, tline1, tline2, tline3, tline4, tline5, tline6, attack_cbp, pass_cbp, cross_cbp, dribble_cbp, shots_cbp,
           score_cbp, seizure, diffence, save_cbp, excepted_goal, shots, shots_success, on_target, pk, passes, pass_success, cross, cross_success, d_fk, i_fk, ck, throwin, throwin_success, dribble, dribble_success, tackle, tackle_success, clear, intercept, offside, yellow, red, app_30, app_penalty, attack_num, chances,  control, agi, kagi]
''' with open('fcryukyu_stats_.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(165):
        test = []
        for j in range(len(header)):
            test.append(comment[j][i])
        writer.writerow(test) '''
print(header)
