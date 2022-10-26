import pandas as pd
import ortega_2022.ortega

samplein = 10
bigdata = pd.read_csv('/Users/rongxiangsu/rongxiang_jupyter/_data/CHTS_interaction/GpsPoints_'+str(samplein)+'min.csv',sep=',', header=0)
bigdata = bigdata[(bigdata['pid']==298473101)|(bigdata['pid']==719939301)]

print(bigdata.head())

