import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extractFileInfo(logfile):
    hand_right = []
    hand_left = []
    handtip_right = []
    handtip_left = []
    thumb_right = []
    thumb_left = []
    shoulder_right = []
    shoulder_left = []
    elbow_right = []
    elbow_left = []
    wrist_right = []
    wrist_left = []
    with open(logfile) as f:
        for line in f:
            if 'Current Timestamp' in line:
                continue
            start = line.index('CI: ') + len('CI: ')
            ci = int(line[start:start + 1])
            start = line.index('Position:(') + len('Position:(')
            temp = line[start:-2].split(',')
            if 'HandRight' in line:
                hand_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'HandLeft' in line:
                hand_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'HandTipRight' in line:
                handtip_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'HandTipLeft' in line:
                handtip_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ThumbRight' in line:
                thumb_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ThumbLeft' in line:
                thumb_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ShoulderRight' in line:
                shoulder_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ShoulderLeft' in line:
                shoulder_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ElbowRight' in line:
                elbow_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'ElbowLeft' in line:
                elbow_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'WristRight' in line:
                wrist_right.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
            elif 'WristLeft' in line:
                wrist_left.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
    return hand_right, hand_left, handtip_right, handtip_left, thumb_right, thumb_left, shoulder_right, shoulder_left, elbow_right, elbow_left, wrist_right, wrist_left


filename = r'C:\Users\Administrator\Desktop\手部识别\HandGrasp.log'
data_grasp = extractFileInfo(filename)
hand_right_grasp, hand_left_grasp, handtip_right_grasp, handtip_left_grasp, thumb_right_grasp, thumb_left_grasp, shoulder_right_grasp, shoulder_left_grasp, elbow_right_grasp, elbow_left_grasp, wrist_right_grasp, wrist_left_grasp = data_grasp

filename = r'C:\Users\Administrator\Desktop\手部识别\HandOpen.log'
data_open = extractFileInfo(filename)
hand_right_open, hand_left_open, handtip_right_open, handtip_left_open, thumb_right_open, thumb_left_open, shoulder_right_open, shoulder_left_open, elbow_right_open, elbow_left_open, wrist_right_open, wrist_left_open = data_open

filename = r'C:\Users\Administrator\Desktop\手部识别\HandClose.log'
data_close = extractFileInfo(filename)
hand_right_close, hand_left_close, handtip_right_close, handtip_left_close, thumb_right_close, thumb_left_close, shoulder_right_close, shoulder_left_close, elbow_right_close, elbow_left_close, wrist_right_close, wrist_left_close = data_close


def calDist(data1, data2):
    dist = []
    for i in range(len(data1)):
        ci1, x1, y1, z1 = data1[i]
        ci2, x2, y2, z2 = data2[i]
        xp = ((x1 - x2) ** 2.0)
        yp = ((y1 - y2) ** 2.0)
        zp = ((z1 - z2) ** 2.0)
        dist.append((xp + yp + zp) ** 0.5)
    return dist


# 绘制折线图
def drawChangePlot(data):
    hand_right, hand_left, handtip_right, handtip_left, thumb_right, thumb_left, shoulder_right, shoulder_left, elbow_right, elbow_left, wrist_right, wrist_left = data
    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
    fig.set_size_inches(15, 5)
    dist = calDist(handtip_right, thumb_right)
    axs[0, 0].plot(dist)
    axs[0, 0].set_title('Right HandTip-Thumb')
    dist = calDist(handtip_right, hand_right)
    axs[0, 1].plot(dist)
    axs[0, 1].set_title('Right HandTip-Hand')
    dist = calDist(handtip_right, wrist_right)
    axs[0, 2].plot(dist)
    axs[0, 2].set_title('Right HandTip-Wrist')
    dist = calDist(thumb_right, hand_right)
    axs[0, 3].plot(dist)
    axs[0, 3].set_title('Right Thumb-Hand')
    dist = calDist(thumb_right, wrist_right)
    axs[0, 4].plot(dist)
    axs[0, 4].set_title('Right Thumb-Wrist')
    dist = calDist(hand_right, wrist_right)
    axs[0, 5].plot(dist)
    axs[0, 5].set_title('Right Hand-Wrist')
    dist = calDist(handtip_left, thumb_left)
    axs[1, 0].plot(dist)
    axs[1, 0].set_title('Left HandTip-Thumb')
    dist = calDist(handtip_left, hand_left)
    axs[1, 1].plot(dist)
    axs[1, 1].set_title('Left HandTip-Hand')
    dist = calDist(handtip_left, wrist_left)
    axs[1, 2].plot(dist)
    axs[1, 2].set_title('Left HandTip-Wrist')
    dist = calDist(thumb_left, hand_left)
    axs[1, 3].plot(dist)
    axs[1, 3].set_title('Left Thumb-Hand')
    dist = calDist(thumb_left, wrist_left)
    axs[1, 4].plot(dist)
    axs[1, 4].set_title('Left Thumb-Wrist')
    dist = calDist(hand_left, wrist_left)
    axs[1, 5].plot(dist)
    axs[1, 5].set_title('Left Hand-Wrist')
    plt.show()


drawChangePlot(data_grasp)
drawChangePlot(data_open)
drawChangePlot(data_close)

# 绘制箱线图
fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
fig.set_size_inches(15, 5)
dist_grasp = calDist(handtip_right_grasp, thumb_right_grasp)
dist_open = calDist(handtip_right_open, thumb_right_open)
dist_close = calDist(handtip_right_close, thumb_right_close)
axs[0, 0].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 0].set_title('Right HandTip-Thumb')
axs[0, 0].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(handtip_right_grasp, hand_right_grasp)
dist_open = calDist(handtip_right_open, hand_right_open)
dist_close = calDist(handtip_right_close, hand_right_close)
axs[0, 1].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 1].set_title('Right HandTip-Hand')
axs[0, 1].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(handtip_right_grasp, wrist_right_grasp)
dist_open = calDist(handtip_right_open, wrist_right_open)
dist_close = calDist(handtip_right_close, wrist_right_close)
axs[0, 2].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 2].set_title('Right HandTip-Wrist')
axs[0, 2].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(thumb_right_grasp, hand_right_grasp)
dist_open = calDist(thumb_right_open, hand_right_open)
dist_close = calDist(thumb_right_close, hand_right_close)
axs[0, 3].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 3].set_title('Right Thumb-Hand')
axs[0, 3].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(thumb_right_grasp, wrist_right_grasp)
dist_open = calDist(thumb_right_open, wrist_right_open)
dist_close = calDist(thumb_right_close, wrist_right_close)
axs[0, 4].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 4].set_title('Right Thumb-Wrist')
axs[0, 4].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(hand_right_grasp, wrist_right_grasp)
dist_open = calDist(hand_right_open, wrist_right_open)
dist_close = calDist(hand_right_close, wrist_right_close)
axs[0, 5].boxplot([dist_grasp, dist_open, dist_close])
axs[0, 5].set_title('Right Hand-Wrist')
axs[0, 5].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(handtip_left_grasp, thumb_left_grasp)
dist_open = calDist(handtip_left_open, thumb_left_open)
dist_close = calDist(handtip_left_close, thumb_left_close)
axs[1, 0].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 0].set_title('Left HandTip-Thumb')
axs[1, 0].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(handtip_left_grasp, hand_left_grasp)
dist_open = calDist(handtip_left_open, hand_left_open)
dist_close = calDist(handtip_left_close, hand_left_close)
axs[1, 1].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 1].set_title('Left HandTip-Hand')
axs[1, 1].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(handtip_left_grasp, wrist_left_grasp)
dist_open = calDist(handtip_left_open, wrist_left_open)
dist_close = calDist(handtip_left_close, wrist_left_close)
axs[1, 2].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 2].set_title('Left HandTip-Wrist')
axs[1, 2].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(thumb_left_grasp, hand_left_grasp)
dist_open = calDist(thumb_left_open, hand_left_open)
dist_close = calDist(thumb_left_close, hand_left_close)
axs[1, 3].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 3].set_title('Left Thumb-Hand')
axs[1, 3].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(thumb_left_grasp, wrist_left_grasp)
dist_open = calDist(thumb_left_open, wrist_left_open)
dist_close = calDist(thumb_left_close, wrist_left_close)
axs[1, 4].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 4].set_title('Left Thumb-Wrist')
axs[1, 4].set_xticklabels(['grasp', 'open', 'close'])
dist_grasp = calDist(hand_left_grasp, wrist_left_grasp)
dist_open = calDist(hand_left_open, wrist_left_open)
dist_close = calDist(hand_left_close, wrist_left_close)
axs[1, 5].boxplot([dist_grasp, dist_open, dist_close])
axs[1, 5].set_title('Left Hand-Wrist')
axs[1, 5].set_xticklabels(['grasp', 'open', 'close'])
plt.show()

# 查看分布图
fig, axs = plt.subplots(2, 6)
fig.set_size_inches(15, 5)
dist_open = calDist(handtip_right_open, thumb_right_open)
dist_close = calDist(handtip_right_close, thumb_right_close)
dist_grasp = calDist(handtip_right_grasp, thumb_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 0], label='open')
sns.kdeplot(dist_close, ax=axs[0, 0], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 0], label='grasp')
axs[0, 0].set_title('Right HandTip-Thumb')
axs[0, 0].legend()
dist_open = calDist(handtip_right_open, hand_right_open)
dist_close = calDist(handtip_right_close, hand_right_close)
dist_grasp = calDist(handtip_right_grasp, hand_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 1], label='open')
sns.kdeplot(dist_close, ax=axs[0, 1], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 1], label='grasp')
axs[0, 1].set_title('Right HandTip-Hand')
axs[0, 1].legend()
dist_open = calDist(handtip_right_open, wrist_right_open)
dist_close = calDist(handtip_right_close, wrist_right_close)
dist_grasp = calDist(handtip_right_grasp, wrist_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 2], label='open')
sns.kdeplot(dist_close, ax=axs[0, 2], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 2], label='grasp')
axs[0, 2].set_title('Right HandTip-Wrist')
axs[0, 2].legend()
dist_open = calDist(thumb_right_open, hand_right_open)
dist_close = calDist(thumb_right_close, hand_right_close)
dist_grasp = calDist(thumb_right_grasp, hand_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 3], label='open')
sns.kdeplot(dist_close, ax=axs[0, 3], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 3], label='grasp')
axs[0, 3].set_title('Right Thumb-Hand')
axs[0, 3].legend()
dist_open = calDist(thumb_right_open, wrist_right_open)
dist_close = calDist(thumb_right_close, wrist_right_close)
dist_grasp = calDist(thumb_right_grasp, wrist_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 4], label='open')
sns.kdeplot(dist_close, ax=axs[0, 4], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 4], label='grasp')
axs[0, 4].set_title('Right Thumb-Wrist')
axs[0, 4].legend()
dist_open = calDist(hand_right_open, wrist_right_open)
dist_close = calDist(hand_right_close, wrist_right_close)
dist_grasp = calDist(hand_right_grasp, wrist_right_grasp)
sns.kdeplot(dist_open, ax=axs[0, 5], label='open')
sns.kdeplot(dist_close, ax=axs[0, 5], label='close')
sns.kdeplot(dist_grasp, ax=axs[0, 5], label='grasp')
axs[0, 5].set_title('Right Hand-Wrist')
axs[0, 5].legend()
dist_open = calDist(handtip_left_open, thumb_left_open)
dist_close = calDist(handtip_left_close, thumb_left_close)
dist_grasp = calDist(handtip_left_grasp, thumb_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 0], label='open')
sns.kdeplot(dist_close, ax=axs[1, 0], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 0], label='grasp')
axs[1, 0].set_title('Left HandTip-Thumb')
axs[1, 0].legend()
dist_open = calDist(handtip_left_open, hand_left_open)
dist_close = calDist(handtip_left_close, hand_left_close)
dist_grasp = calDist(handtip_left_grasp, hand_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 1], label='open')
sns.kdeplot(dist_close, ax=axs[1, 1], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 1], label='grasp')
axs[1, 1].set_title('Left HandTip-Hand')
axs[1, 1].legend()
dist_open = calDist(handtip_left_open, wrist_left_open)
dist_close = calDist(handtip_left_close, wrist_left_close)
dist_grasp = calDist(handtip_left_grasp, wrist_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 2], label='open')
sns.kdeplot(dist_close, ax=axs[1, 2], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 2], label='grasp')
axs[1, 2].set_title('Left HandTip-Wrist')
axs[1, 2].legend()
dist_open = calDist(thumb_left_open, hand_left_open)
dist_close = calDist(thumb_left_close, hand_left_close)
dist_grasp = calDist(thumb_left_grasp, hand_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 3], label='open')
sns.kdeplot(dist_close, ax=axs[1, 3], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 3], label='grasp')
axs[1, 3].set_title('Left Thumb-Hand')
axs[1, 3].legend()
dist_open = calDist(thumb_left_open, wrist_left_open)
dist_close = calDist(thumb_left_close, wrist_left_close)
dist_grasp = calDist(thumb_left_grasp, wrist_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 4], label='open')
sns.kdeplot(dist_close, ax=axs[1, 4], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 4], label='grasp')
axs[1, 4].set_title('Left Thumb-Wrist')
axs[1, 4].legend()
dist_open = calDist(hand_left_open, wrist_left_open)
dist_close = calDist(hand_left_close, wrist_left_close)
dist_grasp = calDist(hand_left_grasp, wrist_left_grasp)
sns.kdeplot(dist_open, ax=axs[1, 5], label='open')
sns.kdeplot(dist_close, ax=axs[1, 5], label='close')
sns.kdeplot(dist_grasp, ax=axs[1, 5], label='grasp')
axs[1, 5].set_title('Left Hand-Wrist')
axs[1, 5].legend()
plt.show()


