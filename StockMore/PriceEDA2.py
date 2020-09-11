import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 再查询2015-2019年，各个上市股票的正常交易数据中的开盘价均价的全量数据
sql = '''
SELECT year, code, AVG(open) AS open_avg
FROM(
	SELECT LEFT(kd.date, 4) AS year, kd.code AS code, kd.open AS open
	FROM history_k_data AS kd
	LEFT JOIN stock_basic AS sb
	ON sb.code = kd.code
	WHERE kd.tradestatus = 1
	AND sb.type = 1
	AND sb.status = 1
) AS temp
GROUP BY year, code
'''
df = pd.read_sql(sql=sql, con=engine)

# 分离各个年度的开盘价均值
df_2015 = df[df['year'] == '2015']
df_2016 = df[df['year'] == '2016']
df_2017 = df[df['year'] == '2017']
df_2018 = df[df['year'] == '2018']
df_2019 = df[df['year'] == '2019']

# 获取2015年开盘价均值的前五名和后五名股票代码
df_2015 = df_2015.sort_values(['open_avg'], ascending=False)
code_top5 = list(df_2015.code)[0:5]
code_last5 = list(df_2015.code)[-5:]

# 绘制前5名股票走势
plt.subplot(1, 2, 1)
for i in range(len(code_top5)):
    top_openavg = [float(df_2015.open_avg[df_2015.code == code_top5[i]]),
                   float(df_2016.open_avg[df_2016.code == code_top5[i]]),
                   float(df_2017.open_avg[df_2017.code == code_top5[i]]),
                   float(df_2018.open_avg[df_2018.code == code_top5[i]]),
                   float(df_2019.open_avg[df_2019.code == code_top5[i]])]
    plt.plot(top_openavg, label='top' + str(i+1) + ' - ' + code_top5[i])
plt.gca().set_xticklabels(["", "2015", "", "2016", "", "2017", "", "2018", "", "2019"])
plt.title("Trend of Stocks whose Open Price is Top at 2015")
plt.legend()

# 绘制后5名股票走势
plt.subplot(1, 2, 2)
for i in range(len(code_last5)):
    last_openavg = [float(df_2015.open_avg[df_2015.code == code_last5[i]]),
                   float(df_2016.open_avg[df_2016.code == code_last5[i]]),
                   float(df_2017.open_avg[df_2017.code == code_last5[i]]),
                   float(df_2018.open_avg[df_2018.code == code_last5[i]]),
                   float(df_2019.open_avg[df_2019.code == code_last5[i]])]
    plt.plot(last_openavg, label='last' + str(i+1) + ' - ' + code_last5[i])
plt.gca().set_xticklabels(["", "2015", "", "2016", "", "2017", "", "2018", "", "2019"])
plt.title("Trend of Stocks whose Open Price is Last at 2015")
plt.legend()

plt.show()

