import math

ef = int(input())
ep = int(input())
nf = int(input())
np = int(input())

bunbo = math.sqrt((ef + nf)*(ef + ep))
bunshi = ef
if bunbo == 0:
    bunbo = 999999999
susscore = bunshi / bunbo
print(susscore)