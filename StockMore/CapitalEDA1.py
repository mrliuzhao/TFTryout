import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 查询2015-2019年、所有上市股票的、正常交易数据的全量信息
sql = '''
SELECT code, concat(year,'-',season) AS ys, close_savg, totalShare, (close_savg * totalShare) AS totalCapital
FROM capitalTemp
ORDER BY year, season
'''

df = pd.read_sql(sql=sql, con=engine)

first2015 = df[df.ys == '2015-1']
first2015 = pd.concat([first2015.code, first2015.totalCapital], axis=1)
second2015 = df[df.ys == '2015-2']
second2015 = pd.concat([second2015.code, second2015.totalCapital], axis=1)
temp = pd.merge(first2015, second2015, on='code', suffixes=('-1','-2'))

third2015 = df[df.ys == '2015-3']
third2015 = pd.concat([third2015.code, third2015.totalCapital], axis=1)
temp = pd.merge(temp, third2015, on='code', suffixes=('', '-3'))

four2015 = df[df.ys == '2015-4']
four2015 = pd.concat([four2015.code, four2015.totalCapital], axis=1)
temp = pd.merge(temp, four2015, on='code', suffixes=('', '-4'))

count = 5
for i in range(4):
    y = str(2016 + i)
    first = df[df.ys == (y + '-1')]
    first = pd.concat([first.code, first.totalCapital], axis=1)
    temp = pd.merge(temp, first, on='code', suffixes=('', '-'+str(count)))
    count += 1
    second = df[df.ys == (y + '-2')]
    second = pd.concat([second.code, second.totalCapital], axis=1)
    temp = pd.merge(temp, second, on='code', suffixes=('', '-'+str(count)))
    count += 1
    third = df[df.ys == (y + '-3')]
    third = pd.concat([third.code, third.totalCapital], axis=1)
    temp = pd.merge(temp, third, on='code', suffixes=('', '-'+str(count)))
    count += 1
    four = df[df.ys == (y + '-4')]
    four = pd.concat([four.code, four.totalCapital], axis=1)
    temp = pd.merge(temp, four, on='code', suffixes=('', '-'+str(count)))
    count += 1

tempT = temp.T.iloc[1:]
tempT.columns = list(temp.code)
tempT = tempT.apply(lambda x: x.astype(float))
cap_cor = tempT.corr(method='pearson')

plt.figure(figsize=(20, 20))
sns.heatmap(cap_cor.iloc[:100, :100], fmt='.2f')

plt.show()

rowcount = 1
corlist = []
for index, row in cap_cor.iterrows():
    corlist.extend(list(row)[rowcount:])
    rowcount += 1

plt.figure(figsize=(16, 9))
plt.hist(corlist, bins=100)
plt.title('Histogram of Correlations between Stock total capitals')
plt.show()


