import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 贵州茅台、山西汾酒、青岛啤酒
stocks = ['sh.600519', 'sh.600809', 'sh.600600']
opendic = {}
for s in stocks:
    sql = '''
    SELECT LEFT(kd.date, 4) AS year, kd.date, kd.code, kd.open, kd.high, kd.low, kd.close, kd.preclose,
    kd.volume, kd.amount, kd.turn, kd.pctChg, kd.peTTM, kd.pbMRQ, kd.psTTM, kd.pcfNcfTTM
    FROM (
        SELECT * FROM history_k_data
        WHERE tradestatus = 1
        AND code = '%s'
    ) AS kd
    LEFT JOIN stock_basic AS sb
    ON sb.code = kd.code
    WHERE sb.type = 1
    AND sb.status = 1
    ''' % s
    df = pd.read_sql(sql=sql, con=engine)
    df_2019 = df[df['year'] == '2019']
    open_2019 = np.log(df_2019.open.dropna())
    opendic[s] = list(open_2019)
    # opendic[s] = open_2019

#  先绘制5年开盘价的箱线图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.figure(figsize=(16, 9))
# plt.boxplot((opendic['sh.600519'], opendic['sh.600809'], opendic['sh.600600']),
#             labels=["贵州茅台", "山西汾酒", "青岛啤酒"], showmeans=True, meanline=True)
# plt.title("开盘价对数箱线图")
# plt.show()

# opendic['sh.600519'].plot.kde()
# opendic['sh.600809'].plot.kde()
# opendic['sh.600600'].plot.kde()
# plt.title("开盘价对数密度图")
# plt.legend(["贵州茅台", "山西汾酒", "青岛啤酒"])
# plt.show()

plt.plot(opendic['sh.600519'])
plt.plot(opendic['sh.600809'])
plt.plot(opendic['sh.600600'])
plt.xticks([])
plt.legend(["贵州茅台", "山西汾酒", "青岛啤酒"])
plt.title("开盘价对数折线图")
plt.show()

