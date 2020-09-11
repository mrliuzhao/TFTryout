import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import datetime

def data_preprocess(stockcode, days_stat):
    connect_info = 'mysql+pymysql://admin:GDCCAdmin.2019@192.168.20.2:3306/stock?charset=utf8'
    engine = create_engine(connect_info)

    # 查询上市股票的正常交易数据的全量数据
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

    # 要统计的涨跌天数
    raise_prob = []
    chg_avg = []
    for index, row in df.iterrows():
        dt_obj = datetime.datetime.strptime(row['date'], "%Y-%m-%d")
        chgs = []
        while len(chgs) < days_stat:
            dt_obj += datetime.timedelta(days=1)
            temp = df[df['date'] == dt_obj.strftime("%Y-%m-%d")]
            if not temp.empty:
                chgs.append(temp['pctChg'].iloc[0])
            if dt_obj > datetime.datetime.now():
                break

        # 统计上涨天数占比
        if len(chgs) == days_stat:
            raise_prob.append(sum([x > 0 for x in chgs]) / days_stat)
            chg_avg.append(sum(chgs) / days_stat)
        else:
            raise_prob.append(np.nan)
            chg_avg.append(np.nan)

    matdf = pd.DataFrame({'raise_prob': raise_prob,
                          'chg_avg': chg_avg})
    outputdf = pd.concat([df, matdf], axis=1)
    outputdf.to_csv(r'.\data\%s-%02dd-raiseprob.csv' % (stockcode, days_stat))


if __name__ == '__main__':
    stockcodes = ['sz.000063', 'sz.300463', 'sh.601318',
                  'sh.600519', 'sz.000651']
    for stockcode in stockcodes:
        data_preprocess(stockcode, 10)


