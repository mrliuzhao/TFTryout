import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import datetime

connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
engine = create_engine(connect_info)

# 查询上市股票的正常交易数据的全量数据
# stockcode = 'sz.300463'
# stockcode = 'sh.601318'
# stockcode = 'sz.000063'
stockcode = 'sz.000651'
# stockcode = 'sh.600519'
sql = '''
SELECT kd.date, kd.code, kd.open, kd.high, kd.low, kd.close, kd.preclose,
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
ORDER BY date
''' % stockcode

df = pd.read_sql(sql=sql, con=engine)

# 选择最起码7天后的首个交易日的价格作为预测目标
days_delays = [1, 3, 7, 15]
for dd in days_delays:
    delay_date = []
    open_delay = []
    close_delay = []
    high_delay = []
    low_delay = []
    chg_delay = []
    lastdate = None
    for index, row in df.iterrows():
        dt_obj = datetime.datetime.strptime(row['date'], "%Y-%m-%d")
        dt_obj += datetime.timedelta(days=dd)
        if lastdate is not None and lastdate > dt_obj:
            date_base = lastdate
        else:
            date_base = dt_obj

        find_date = False
        # 最多再多查看100天后的首个交易日
        for i in range(101):
            temp_date = date_base + datetime.timedelta(days=i)
            if temp_date > datetime.datetime.now():
                break
            else:
                temp = df[df['date'] == temp_date.strftime("%Y-%m-%d")]
                if not temp.empty:
                    find_date = True
                    lastdate = temp_date + datetime.timedelta(days=1)
                    open_delay.append(temp['open'].iloc[0])
                    close_delay.append(temp['close'].iloc[0])
                    high_delay.append(temp['high'].iloc[0])
                    low_delay.append(temp['low'].iloc[0])
                    chg_delay.append(temp['pctChg'].iloc[0])
                    delay_date.append(temp_date.strftime("%Y-%m-%d"))
                    break
        if not find_date:
            # lastdate = None
            open_delay.append(np.nan)
            close_delay.append(np.nan)
            high_delay.append(np.nan)
            low_delay.append(np.nan)
            chg_delay.append(np.nan)
            delay_date.append(np.nan)

    matdf = pd.DataFrame({'open_delay': open_delay,
                          'close_delay': close_delay,
                          'high_delay': high_delay,
                          'low_delay': low_delay,
                          'pctChg_delay': chg_delay,
                          'date_delay': delay_date})
    outputdf = pd.concat([df, matdf], axis=1)
    # outputdf.to_csv(r'.\data\%s-7d.csv' % stockcode)
    # outputdf.to_csv(r'.\data\%s-3d.csv' % stockcode)
    outputdf.to_csv(r'.\data\%s-%02dd.csv' % (stockcode, dd))

