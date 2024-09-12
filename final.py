### DAP round

import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import math
import sklearn.linear_model


### Dataframe reading / cleaning
df = pd.read_csv("DAP-Dataset.csv")

print(df.shape)
    # 9899 * 4
print(df.columns)
df.rename(columns = {'Unnamed: 0' : 'time'}, inplace = True)
column = df.columns
    # column names are 'Unnamed: 0', 'SNPB', 'NVNK', 'ZZZZZ'
print(f'any NANs in the dataframe? { not (df.shape == df.dropna().shape)}')
    # no NANs
print(df.head)


# plt.plot(df[column[0]], df[column[1]])
# plt.title(f'time - {column[1]}')
# plt.show()
# plt.plot(df[column[0]], df[column[2]])
# plt.title(f'time - {column[2]}')
# plt.show()
# plt.plot(df[column[0]], df[column[3]])
# plt.title(f'time - {column[3]}')
# plt.show()
#
# plt.plot(df[column[0]], df[column[1]])
# plt.plot(df[column[0]], df[column[2]])
# plt.plot(df[column[0]], df[column[3]])
# plt.legend()
# plt.title(f'time - all')
# plt.show()


stats = pd.DataFrame([df.mean(), df.std(), df.var()], index=['mean', 'stdev', 'var'])
stats = stats.iloc[:, 1:]
print(stats)
#             SNPB       NVNK      ZZZZZ
# mean   83.235320  46.131258  37.194234
# stdev   0.846474   2.216037   2.245984
# var     0.716519   4.910820   5.044444


corr12 = df[column[1]].corr(df[column[2]])
corr13 = df[column[1]].corr(df[column[3]])
corr23 = df[column[2]].corr(df[column[3]])
print(f'correlation between {column[1]} and {column[2]} :{corr12}')
print(f'correlation between {column[1]} and {column[3]} :{corr13}')
print(f'correlation between {column[2]} and {column[3]} :{corr23}')
# correlation between SNPB and NVNK :0.5180641991745849
# correlation between SNPB and ZZZZZ :0.5500763466500627
# correlation between NVNK and ZZZZZ :0.7789553516716655



### IMPLEMENTING PAIRS TRADING IDEA

time = df['time']
S1 = df['SNPB']
S2 = df['NVNK']
S3 = df['ZZZZZ']

ratio_12 = S1 / S2
ratio_23 = S2 / S3
ratio_31 = S3 / S1

const = 0.001
ratio_12_ewm = ratio_12.ewm(alpha = const)
ratio_12_norm = (ratio_12 - ratio_12_ewm.mean()) / ratio_12_ewm.std()
ratio_23_ewm = ratio_23.ewm(alpha = const)
ratio_23_norm = (ratio_23 - ratio_23_ewm.mean()) / ratio_23_ewm.std()
ratio_31_ewm = ratio_31.ewm(alpha = const)
ratio_31_norm = (ratio_31 - ratio_31_ewm.mean()) / ratio_31_ewm.std()

plt.plot(time, ratio_12_norm)
plt.title(f'time - ratio_12_norm')
plt.show()

plt.plot(time, ratio_23_norm)
plt.title(f'time - ratio_23_norm')
plt.show()

plt.plot(time, ratio_31_norm)
plt.title(f'time - ratio_31_norm')
plt.show()

totaltime = len(df['time'])
buy_cap, sell_cap = 1, 0.5

seed = 1000
s1, s2, s3 = 0, 0, 0

# short is not allowed
for time in range(totaltime):
    if ratio_12_norm[time] > buy_cap:   # buy s2
        seed -= S2[time]
        s2 += 1
    elif ratio_12_norm[time] < -buy_cap:
        seed -= S1[time]
        s1 += 1
    elif abs(ratio_12_norm[time]) < sell_cap:
        seed += S1[time]*s1 + S2[time]*s2
        s1, s2 = 0, 0

    if ratio_23_norm[time] > buy_cap:   # buy s3
        seed -= S3[time]
        s3 += 1
    elif ratio_23_norm[time] < -buy_cap:
        seed -= S2[time]
        s2 += 1
    elif abs(ratio_23_norm[time]) < sell_cap:
        seed += S2[time]*s2 + S3[time]*s3
        s2, s3 = 0, 0

    if ratio_31_norm[time] > buy_cap:   # buy s1
        seed -= S1[time]
        s1 += 1
    elif ratio_31_norm[time] < -buy_cap:
        seed -= S3[time]
        s3 += 1
    elif abs(ratio_31_norm[time]) < sell_cap:
        seed += S3[time]*s3 + S1[time]*s1
        s3, s1 = 0, 0
seed += S1[totaltime-1]*s1 + S2[totaltime-1]*s2 + S3[totaltime-1]*s3
print(f'{seed: .2f}')
# 1000 -> 1293.74


# Several ideas can develop the P&L such as
# when to buy/sell
# adjusting the amount of stocks at the buying signal
# changing the window sizes






