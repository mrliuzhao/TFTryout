import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 先计算每个公司、每个季度的平均收盘价，再乘以该季度该公司的总股本，计算每个公司每季度总资本
# 查询2015-2019年、所有上市股票的季度总资本
sql = '''
SELECT p.code, p.year, p.season, (p.close_savg * s.totalShare) AS totalCapital FROM (
	SELECT kd.code AS code, LEFT(kd.date, 4) AS year, ceil(cast(substr(kd.date,6,2) AS SIGNED) / 3) AS season, AVG(kd.close) AS close_savg
	FROM history_k_data AS kd
	LEFT JOIN stock_basic AS sb
	ON sb.code = kd.code
	WHERE kd.tradestatus = 1
	AND sb.type = 1
	AND sb.status = 1
	AND cast(LEFT(kd.date,4) AS SIGNED) >= 2015
	AND cast(LEFT(kd.date,4) AS SIGNED) <= 2019
	GROUP BY code, year, season
) AS p
LEFT JOIN (
	SELECT code, LEFT(statDate, 4) AS year, ceil(cast(substr(statDate,6,2) AS SIGNED) / 3) AS season, totalShare 
	FROM season_profit
) AS s
ON p.code = s.code
AND p.year = s.year
AND p.season = s.season
'''
df = pd.read_sql(sql=sql, con=engine)
df_2015 = df[df['year'] == '2015']
df_2016 = df[df['year'] == '2016']
df_2017 = df[df['year'] == '2017']
df_2018 = df[df['year'] == '2018']
df_2019 = df[df['year'] == '2019']
capital_2015 = df_2015.totalCapital.dropna()
capital_2016 = df_2016.totalCapital.dropna()
capital_2017 = df_2017.totalCapital.dropna()
capital_2018 = df_2018.totalCapital.dropna()
capital_2019 = df_2019.totalCapital.dropna()

#  绘制5年总资本箱线图
plt.figure(figsize=(16, 9))
plt.boxplot((capital_2015, capital_2016, capital_2017,
             capital_2018, capital_2019),
            labels=["2015", "2016", "2017", "2018", "2019"])
plt.title("Total Capital Summary")
plt.show()

# 绘制5年总资本直方图
plt.figure(figsize=(16, 9))
plt.subplot(2, 3, 1)
plt.hist(capital_2015, bins=100)
plt.title('2015 Total Capitals')

plt.subplot(2, 3, 2)
plt.hist(capital_2016, bins=100)
plt.title('2016 Total Capitals')

plt.subplot(2, 3, 3)
plt.hist(capital_2017, bins=100)
plt.title('2017 Total Capitals')

plt.subplot(2, 3, 4)
plt.hist(capital_2018, bins=100)
plt.title('2018 Total Capitals')

plt.subplot(2, 3, 5)
plt.hist(capital_2019, bins=100)
plt.title('2019 Total Capitals')
plt.show()


# 获取2015年总资本的前10名和后10名股票代码
cap2015_avg = df_2015.groupby(['code']).mean()
cap2015_avg = cap2015_avg.sort_values(['totalCapital'], ascending=False)
code_top = list(cap2015_avg.totalCapital.keys())[0:10]
code_last = list(cap2015_avg.totalCapital.keys())[-10:]

cap2016_avg = df_2016.groupby(['code']).mean()
cap2017_avg = df_2017.groupby(['code']).mean()
cap2018_avg = df_2018.groupby(['code']).mean()
cap2019_avg = df_2019.groupby(['code']).mean()
plt.figure(figsize=(16, 18))
# 绘制前10名股票走势
plt.subplot(2, 1, 1)
for i in range(len(code_top)):
    top_capavg = [cap2015_avg.totalCapital[code_top[i]],
                  cap2016_avg.totalCapital[code_top[i]],
                  cap2017_avg.totalCapital[code_top[i]],
                  cap2018_avg.totalCapital[code_top[i]],
                  cap2019_avg.totalCapital[code_top[i]]]
    plt.plot(top_capavg, label='top' + str(i+1) + ' - ' + code_top[i])
plt.gca().set_xticklabels(["", "2015", "", "2016", "", "2017", "", "2018", "", "2019"])
plt.title("Trend of Stocks whose Total Capital is Top at 2015")
plt.legend()

# 绘制后10名股票走势
plt.subplot(2, 1, 2)
for i in range(len(code_last)):
    last_topavg = [cap2015_avg.totalCapital[code_last[i]],
                   cap2016_avg.totalCapital[code_last[i]],
                   cap2017_avg.totalCapital[code_last[i]],
                   cap2018_avg.totalCapital[code_last[i]],
                   cap2019_avg.totalCapital[code_last[i]]]
    plt.plot(last_topavg, label='last' + str(i+1) + ' - ' + code_last[i])
plt.gca().set_xticklabels(["", "2015", "", "2016", "", "2017", "", "2018", "", "2019"])
plt.title("Trend of Stocks whose Total Capital is Last at 2015")
plt.legend()

plt.show()


# 检测分割点
bounds = np.arange(0.01, 0.5, 0.01)
r2015 = []
r2016 = []
r2017 = []
r2018 = []
r2019 = []
for b in bounds:
    r2015.append(len(capital_2015[capital_2015 <= b * 1e12]) / len(capital_2015))
    r2016.append(len(capital_2016[capital_2016 <= b * 1e12]) / len(capital_2016))
    r2017.append(len(capital_2017[capital_2017 <= b * 1e12]) / len(capital_2017))
    r2018.append(len(capital_2018[capital_2018 <= b * 1e12]) / len(capital_2018))
    r2019.append(len(capital_2019[capital_2019 <= b * 1e12]) / len(capital_2019))

plt.figure(figsize=(8, 6))
plt.plot(bounds, r2015, label='2015')
plt.plot(bounds, r2016, label='2016')
plt.plot(bounds, r2017, label='2017')
plt.plot(bounds, r2018, label='2018')
plt.plot(bounds, r2019, label='2019')
plt.xlabel('bound')
plt.ylabel('low part ratio')
plt.legend()
plt.show()

# 以0.2*1e12为界限，拆分出每年的高资本部分和低资本部分
bound = 0.2 * 1e12
cap_2015_low = list(capital_2015[capital_2015 <= bound])
cap_2015_high = list(capital_2015[capital_2015 > bound])
cap_2016_low = list(capital_2016[capital_2016 <= bound])
cap_2016_high = list(capital_2016[capital_2016 > bound])
cap_2017_low = list(capital_2017[capital_2017 <= bound])
cap_2017_high = list(capital_2017[capital_2017 > bound])
cap_2018_low = list(capital_2018[capital_2018 <= bound])
cap_2018_high = list(capital_2018[capital_2018 > bound])
cap_2019_low = list(capital_2019[capital_2019 <= bound])
cap_2019_high = list(capital_2019[capital_2019 > bound])

# 做出每年高价部分和低价部分的直方图查看分布情况
plt.figure(figsize=(15, 23))
plt.subplot(5, 2, 1)
plt.hist(cap_2015_low, bins=100)
plt.title('2015 Capital Low')
plt.subplot(5, 2, 2)
plt.hist(cap_2015_high, bins=100)
plt.title('2015 Capital High')

plt.subplot(5, 2, 3)
plt.hist(cap_2016_low, bins=100)
plt.title('2016 Capital Low')
plt.subplot(5, 2, 4)
plt.hist(cap_2016_high, bins=100)
plt.title('2016 Capital High')

plt.subplot(5, 2, 5)
plt.hist(cap_2017_low, bins=100)
plt.title('2017 Capital Low')
plt.subplot(5, 2, 6)
plt.hist(cap_2017_high, bins=100)
plt.title('2017 Capital High')

plt.subplot(5, 2, 7)
plt.hist(cap_2018_low, bins=100)
plt.title('2018 Capital Low')
plt.subplot(5, 2, 8)
plt.hist(cap_2018_high, bins=100)
plt.title('2018 Capital High')

plt.subplot(5, 2, 9)
plt.hist(cap_2019_low, bins=100)
plt.title('2019 Capital Low')
plt.subplot(5, 2, 10)
plt.hist(cap_2019_high, bins=100)
plt.title('2019 Capital High')

plt.show()



