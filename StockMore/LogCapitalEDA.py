import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 再查询2015-2019年，各个上市股票的正常交易数据中的总资本的全量数据
sql = '''
SELECT code, year, season, close_savg, totalShare, (close_savg * totalShare) AS totalCapital
FROM capitalTemp
ORDER BY year, season
'''
df = pd.read_sql(sql=sql, con=engine)

# 分离各个年度的开盘价
df_2015 = df[df['year'] == '2015']
df_2016 = df[df['year'] == '2016']
df_2017 = df[df['year'] == '2017']
df_2018 = df[df['year'] == '2018']
df_2019 = df[df['year'] == '2019']
capital_2015 = np.log(df_2015.totalCapital.dropna())
capital_2016 = np.log(df_2016.totalCapital.dropna())
capital_2017 = np.log(df_2017.totalCapital.dropna())
capital_2018 = np.log(df_2018.totalCapital.dropna())
capital_2019 = np.log(df_2019.totalCapital.dropna())

#  先绘制5年总资本的箱线图
plt.figure(figsize=(16, 9))
plt.boxplot((capital_2015, capital_2016, capital_2017, capital_2018, capital_2019),
            labels=["2015", "2016", "2017", "2018", "2019"])
plt.title("Log of Total Capital Boxplot")
plt.show()

# 再绘制这5年总资本的直方图
plt.figure(figsize=(16, 9))
plt.subplot(2, 3, 1)
sns.distplot(capital_2015, kde=True, fit=st.norm)
plt.xlabel('')
plt.title('2015 Total Capitals')

plt.subplot(2, 3, 2)
sns.distplot(capital_2016, kde=True, fit=st.norm)
plt.xlabel('')
plt.title('2016 Total Capitals')

plt.subplot(2, 3, 3)
sns.distplot(capital_2017, kde=True, fit=st.norm)
plt.xlabel('')
plt.title('2017 Total Capitals')

plt.subplot(2, 3, 4)
sns.distplot(capital_2018, kde=True, fit=st.norm)
plt.xlabel('')
plt.title('2018 Total Capitals')

plt.subplot(2, 3, 5)
sns.distplot(capital_2019, kde=True, fit=st.norm)
plt.xlabel('')
plt.title('2019 Total Capitals')
plt.show()

des2015 = list(capital_2015.describe())
des2015.append(capital_2015.mode()[0])
des2015.append(capital_2015.skew())
des2015.append(capital_2015.kurt())
des2016 = list(capital_2016.describe())
des2016.append(capital_2016.mode()[0])
des2016.append(capital_2016.skew())
des2016.append(capital_2016.kurt())
des2017 = list(capital_2017.describe())
des2017.append(capital_2017.mode()[0])
des2017.append(capital_2017.skew())
des2017.append(capital_2017.kurt())
des2018 = list(capital_2018.describe())
des2018.append(capital_2018.mode()[0])
des2018.append(capital_2018.skew())
des2018.append(capital_2018.kurt())
des2019 = list(capital_2019.describe())
des2019.append(capital_2019.mode()[0])
des2019.append(capital_2019.skew())
des2019.append(capital_2019.kurt())
statsDF = pd.DataFrame({'2015 Capital Stats': list(des2015),
                     '2016 Capital Stats': list(des2016),
                     '2017 Capital Stats': list(des2017),
                     '2018 Capital Stats': list(des2018),
                     '2019 Capital Stats': list(des2019)})
statsDF.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%',
                 'max', 'mode', 'skewness', 'kurtosis']

statsDF.iloc[1, :].plot.line()
statsDF.iloc[5, :].plot.line()
ax = statsDF.iloc[8, :].plot.line()
ax.set_xticklabels(['', '2015', '', '2016', '', '2017',
                    '', '2018', '', '2019'])
plt.legend(['mean', 'median', 'mode'])
plt.show()



