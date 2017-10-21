import pandas
from pandas.io import data

with open("rt-polarity.txt")as fo:
    for i in fo:
        df=pandas.DataFrame(data[i])
        df.to_csv('ok.csv', mode='a', encoding='utf-8', index=False)