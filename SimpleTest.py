import matplotlib.pyplot as plt
import pandas as pd

# filename = r'C:\Users\Administrator\Desktop\logs(67)\2020-09-14\fes-main-0.log'
# filename = r'C:\Users\Administrator\Desktop\logs(69)\2020-09-14\fes-main-0.log'
#
# timestamps = []
# pitchAngle = []
# rollAngle = []
# yawAngle = []
# with open(filename) as f:
#     for line in f:
#         if '获得头部实时旋转角度信息' in line:
#             start = line.index('TimeStamp')
#             parts = line[start:].split(';')
#             timestamps.append(int(parts[0].split(': ')[1]))
#             pitchAngle.append(float(parts[1].split(': ')[1]))
#             rollAngle.append(float(parts[2].split(': ')[1]))
#             yawAngle.append(float(parts[3].split(': ')[1]))
#
# for i in range(1, len(timestamps)):
#     last = yawAngle[i - 1]
#     cur = yawAngle[i]

df = pd.read_csv(r'.\StockMore\data\sh.600519-01d.csv')

close = list(df['close'])
b1 = 0.9
b2 = 0.99
ma1 = [0.0]
ma2 = [0.0]
for c in close:
    temp1 = b1 * ma1[-1] + (1 - b1) * c
    ma1.append(temp1)
    temp2 = b2 * ma2[-1] + (1 - b2) * c
    ma2.append(temp2)

plt.plot(close, label='original values')
plt.plot(ma1, label='beta = 0.9')
plt.plot(ma2, label='beta = 0.99')
plt.title('Moving Average with different beta')
plt.legend()
plt.show()
