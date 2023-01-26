w = [[0], [0]]
for i in range(10):
    for j in range(10):
        w[0][i] += i
        w[1][j] += j * 5

f = open('lightgbm.txt', 'a')
f.write(
    f'これはテストです{w}')
f.close()
