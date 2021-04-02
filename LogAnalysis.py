import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns


def str2Timestamp(time_str):
    milisec = int(time_str.split('.')[1])
    t1 = time.strptime(time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
    t2 = int(time.mktime(t1))*1000 + milisec
    return t2


def extractFileinfo(logfile):
    jointTimeList = []
    mainTimeStrlist = []
    deltatime = []
    poseCost = []
    handCost = []
    totalCost = []
    rotationCost = []
    with open(logfile) as f:
        for line in f:
            if 'Get Joints, Timestamp: ' in line:
                timeStr = line[:23]
                mainTimeStrlist.append(timeStr)
                milisec = int(timeStr.split('.')[1])
                t1 = time.strptime(timeStr.split('.')[0], '%Y-%m-%d %H:%M:%S')
                mainserviceTimestamp = int(time.mktime(t1)) * 1000 + milisec
                start = line.index('Get Joints, Timestamp: ') + len('Get Joints, Timestamp: ')
                jointTimestamp = int(line[start:])
                jointTimeList.append(jointTimestamp)
                deltatime.append(mainserviceTimestamp - jointTimestamp)
            if 'Detect Pose Fail Time Cost: 'in line:
                start = line.index('Detect Pose Fail Time Cost: ') + len('Detect Pose Fail Time Cost: ')
                poseCost.append(int(line[start:]))
            if 'Detect HandState Time Cost: 'in line:
                start = line.index('Detect HandState Time Cost: ') + len('Detect HandState Time Cost: ')
                handCost.append(int(line[start:]))
            if 'Total Process Time Cost: 'in line:
                start = line.index('Total Process Time Cost: ') + len('Total Process Time Cost: ')
                totalCost.append(int(line[start:]))
            if 'Record Shoulder Rotation Speed Time Cost: ' in line:
                start = line.index('Record Shoulder Rotation Speed Time Cost: ') + len('Record Shoulder Rotation Speed Time Cost: ')
                rotationCost.append(int(line[start:]))
    return mainTimeStrlist, jointTimeList, deltatime, poseCost, handCost, totalCost, rotationCost


filename = r'C:\Users\Administrator\Desktop\logs-mock\2021-01-19\fes-main-0.log'
mainTime_mock, jointTime_mock, deltatime_mock, poseCost_mock, handCost_mock, totalCost_mock, rotationCost_mock = extractFileinfo(filename)
filename = r'C:\Users\Administrator\Desktop\logs-1avg\2021-01-19\fes-main-0.log'
mainTime_1avg, jointTime_1avg, deltatime_1avg, poseCost_1avg, handCost_1avg, totalCost_1avg, rotationCost_1avg = extractFileinfo(filename)
filename = r'C:\Users\Administrator\Desktop\logs-3avg\2021-01-19\fes-main-0.log'
mainTime_3avg, jointTime_3avg, deltatime_3avg, poseCost_3avg, handCost_3avg, totalCost_3avg, rotationCost_3avg = extractFileinfo(filename)

# 查看VRMeds.dll发送关节点和MainService接收关节点之间的时间差
plt.boxplot([deltatime_mock, deltatime_1avg, deltatime_3avg])
plt.title('Delta Times Between Joints Send and Receive')
plt.xticks([1, 2, 3], ['Mock', '1average', '3average'])
plt.show()

# 查看MainService各种计算的耗时
plt.boxplot([poseCost_mock, handCost_mock, rotationCost_mock, totalCost_mock])
plt.title('Time Cost in MainService when Mock')
plt.xticks([1, 2, 3, 4], ['PoseDetect', 'HandDetect', 'RotationCost', 'TotalCost'])
plt.show()

plt.boxplot([poseCost_1avg, handCost_1avg, rotationCost_1avg, totalCost_1avg])
plt.title('Time Cost in MainService when 1 Average')
plt.xticks([1, 2, 3, 4], ['PoseDetect', 'HandDetect', 'RotationCost', 'TotalCost'])
plt.show()

plt.boxplot([poseCost_3avg, handCost_3avg, rotationCost_3avg, totalCost_3avg])
plt.title('Time Cost in MainService when 3 Average')
plt.xticks([1, 2, 3, 4], ['PoseDetect', 'HandDetect', 'RotationCost', 'TotalCost'])
plt.show()

# 过滤超过2秒的数据
def calDeltaTime(timeStamps):
    deltatime = []
    invalidCount = 0
    for i in range(len(timeStamps) - 1):
        dd = timeStamps[i + 1] - timeStamps[i]
        if dd > 2000:
            invalidCount += 1
            continue
        deltatime.append(timeStamps[i + 1] - timeStamps[i])
    return deltatime, invalidCount


dd_mock, invalid_mock = calDeltaTime(jointTime_mock)
dd_1avg, invalid_1avg = calDeltaTime(jointTime_1avg)
dd_3avg, invalid_3avg = calDeltaTime(jointTime_3avg)

plt.boxplot([dd_mock, dd_1avg, dd_3avg])
plt.title('Delta Times Between Two Joints')
plt.xticks([1, 2, 3], ['Mock', '1average', '3average'])
plt.show()

sns.kdeplot(dd_mock, label='Mock')
sns.kdeplot(dd_1avg, label='1average')
sns.kdeplot(dd_3avg, label='3average')
plt.title('Delta Times Between Two Joints')
plt.legend()
plt.show()


