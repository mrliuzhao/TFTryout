import matplotlib.pyplot as plt


# filename = r'C:\Users\Administrator\Desktop\logs(67)\2020-09-14\fes-main-0.log'
filename = r'C:\Users\Administrator\Desktop\logs(69)\2020-09-14\fes-main-0.log'

timestamps = []
pitchAngle = []
rollAngle = []
yawAngle = []
with open(filename) as f:
    for line in f:
        if '获得头部实时旋转角度信息' in line:
            start = line.index('TimeStamp')
            parts = line[start:].split(';')
            timestamps.append(int(parts[0].split(': ')[1]))
            pitchAngle.append(float(parts[1].split(': ')[1]))
            rollAngle.append(float(parts[2].split(': ')[1]))
            yawAngle.append(float(parts[3].split(': ')[1]))

for i in range(1, len(timestamps)):
    last = yawAngle[i - 1]
    cur = yawAngle[i]


