import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

filename = r'C:\Users\Administrator\Desktop\手部识别\HandOpen.log'

hand_right_open = []
hand_left_open = []
handtip_right_open = []
handtip_left_open = []
thumb_right_open = []
thumb_left_open = []
shoulder_right_open = []
shoulder_left_open = []
elbow_right_open = []
elbow_left_open = []
wrist_right_open = []
wrist_left_open = []
with open(filename) as f:
    for line in f:
        if 'Current Timestamp' in line:
            continue
        start = line.index('CI: ') + len('CI: ')
        ci = int(line[start:start + 1])
        start = line.index('Position:(') + len('Position:(')
        temp = line[start:-2].split(',')
        if 'HandRight' in line:
            hand_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandLeft' in line:
            hand_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandTipRight' in line:
            handtip_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandTipLeft' in line:
            handtip_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ThumbRight' in line:
            thumb_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ThumbLeft' in line:
            thumb_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ShoulderRight' in line:
            shoulder_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ShoulderLeft' in line:
            shoulder_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ElbowRight' in line:
            elbow_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ElbowLeft' in line:
            elbow_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'WristRight' in line:
            wrist_right_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'WristLeft' in line:
            wrist_left_open.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))

filename = r'C:\Users\Administrator\Desktop\手部识别\HandClose.log'
hand_right_close = []
hand_left_close = []
handtip_right_close = []
handtip_left_close = []
thumb_right_close = []
thumb_left_close = []
shoulder_right_close = []
shoulder_left_close = []
elbow_right_close = []
elbow_left_close = []
wrist_right_close = []
wrist_left_close = []
with open(filename) as f:
    for line in f:
        if 'Current Timestamp' in line:
            continue
        start = line.index('CI: ') + len('CI: ')
        ci = int(line[start:start + 1])
        start = line.index('Position:(') + len('Position:(')
        temp = line[start:-2].split(',')
        if 'HandRight' in line:
            hand_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandLeft' in line:
            hand_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandTipRight' in line:
            handtip_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'HandTipLeft' in line:
            handtip_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ThumbRight' in line:
            thumb_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ThumbLeft' in line:
            thumb_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ShoulderRight' in line:
            shoulder_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ShoulderLeft' in line:
            shoulder_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ElbowRight' in line:
            elbow_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'ElbowLeft' in line:
            elbow_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'WristRight' in line:
            wrist_right_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))
        elif 'WristLeft' in line:
            wrist_left_close.append((ci, float(temp[0]), float(temp[1]), float(temp[2])))


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


# fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
# fig.set_size_inches(15, 5)
# dist_open = calDist(handtip_right_open, thumb_right_open)
# dist_close = calDist(handtip_right_close, thumb_right_close)
# axs[0, 0].boxplot([dist_open, dist_close])
# axs[0, 0].set_title('Right HandTip-Thumb')
# axs[0, 0].set_xticklabels(['open', 'close'])
# dist_open = calDist(handtip_right_open, hand_right_open)
# dist_close = calDist(handtip_right_close, hand_right_close)
# axs[0, 1].boxplot([dist_open, dist_close])
# axs[0, 1].set_title('Right HandTip-Hand')
# axs[0, 1].set_xticklabels(['open', 'close'])
# dist_open = calDist(handtip_right_open, wrist_right_open)
# dist_close = calDist(handtip_right_close, wrist_right_close)
# axs[0, 2].boxplot([dist_open, dist_close])
# axs[0, 2].set_title('Right HandTip-Wrist')
# axs[0, 2].set_xticklabels(['open', 'close'])
# dist_open = calDist(thumb_right_open, hand_right_open)
# dist_close = calDist(thumb_right_close, hand_right_close)
# axs[0, 3].boxplot([dist_open, dist_close])
# axs[0, 3].set_title('Right Thumb-Hand')
# axs[0, 3].set_xticklabels(['open', 'close'])
# dist_open = calDist(thumb_right_open, wrist_right_open)
# dist_close = calDist(thumb_right_close, wrist_right_close)
# axs[0, 4].boxplot([dist_open, dist_close])
# axs[0, 4].set_title('Right Thumb-Wrist')
# axs[0, 4].set_xticklabels(['open', 'close'])
# dist_open = calDist(hand_right_open, wrist_right_open)
# dist_close = calDist(hand_right_close, wrist_right_close)
# axs[0, 5].boxplot([dist_open, dist_close])
# axs[0, 5].set_title('Right Hand-Wrist')
# axs[0, 5].set_xticklabels(['open', 'close'])
# dist_open = calDist(handtip_left_open, thumb_left_open)
# dist_close = calDist(handtip_left_close, thumb_left_close)
# axs[1, 0].boxplot([dist_open, dist_close])
# axs[1, 0].set_title('Left HandTip-Thumb')
# axs[1, 0].set_xticklabels(['open', 'close'])
# dist_open = calDist(handtip_left_open, hand_left_open)
# dist_close = calDist(handtip_left_close, hand_left_close)
# axs[1, 1].boxplot([dist_open, dist_close])
# axs[1, 1].set_title('Left HandTip-Hand')
# axs[1, 1].set_xticklabels(['open', 'close'])
# dist_open = calDist(handtip_left_open, wrist_left_open)
# dist_close = calDist(handtip_left_close, wrist_left_close)
# axs[1, 2].boxplot([dist_open, dist_close])
# axs[1, 2].set_title('Left HandTip-Wrist')
# axs[1, 2].set_xticklabels(['open', 'close'])
# dist_open = calDist(thumb_left_open, hand_left_open)
# dist_close = calDist(thumb_left_close, hand_left_close)
# axs[1, 3].boxplot([dist_open, dist_close])
# axs[1, 3].set_title('Left Thumb-Hand')
# axs[1, 3].set_xticklabels(['open', 'close'])
# dist_open = calDist(thumb_left_open, wrist_left_open)
# dist_close = calDist(thumb_left_close, wrist_left_close)
# axs[1, 4].boxplot([dist_open, dist_close])
# axs[1, 4].set_title('Left Thumb-Wrist')
# axs[1, 4].set_xticklabels(['open', 'close'])
# dist_open = calDist(hand_left_open, wrist_left_open)
# dist_close = calDist(hand_left_close, wrist_left_close)
# axs[1, 5].boxplot([dist_open, dist_close])
# axs[1, 5].set_title('Left Hand-Wrist')
# axs[1, 5].set_xticklabels(['open', 'close'])
# plt.show()


# 查看分布图
fig, axs = plt.subplots(2, 6)
fig.set_size_inches(15, 5)
dist_open = calDist(handtip_right_open, thumb_right_open)
dist_close = calDist(handtip_right_close, thumb_right_close)
sns.kdeplot(dist_open, ax=axs[0, 0], label='open')
sns.kdeplot(dist_close, ax=axs[0, 0], label='close')
axs[0, 0].set_title('Right HandTip-Thumb')
axs[0, 0].legend()
dist_open = calDist(handtip_right_open, hand_right_open)
dist_close = calDist(handtip_right_close, hand_right_close)
sns.kdeplot(dist_open, ax=axs[0, 1], label='open')
sns.kdeplot(dist_close, ax=axs[0, 1], label='close')
axs[0, 1].set_title('Right HandTip-Hand')
axs[0, 1].legend()
dist_open = calDist(handtip_right_open, wrist_right_open)
dist_close = calDist(handtip_right_close, wrist_right_close)
sns.kdeplot(dist_open, ax=axs[0, 2], label='open')
sns.kdeplot(dist_close, ax=axs[0, 2], label='close')
axs[0, 2].set_title('Right HandTip-Wrist')
axs[0, 2].legend()
dist_open = calDist(thumb_right_open, hand_right_open)
dist_close = calDist(thumb_right_close, hand_right_close)
sns.kdeplot(dist_open, ax=axs[0, 3], label='open')
sns.kdeplot(dist_close, ax=axs[0, 3], label='close')
axs[0, 3].set_title('Right Thumb-Hand')
axs[0, 3].legend()
dist_open = calDist(thumb_right_open, wrist_right_open)
dist_close = calDist(thumb_right_close, wrist_right_close)
sns.kdeplot(dist_open, ax=axs[0, 4], label='open')
sns.kdeplot(dist_close, ax=axs[0, 4], label='close')
axs[0, 4].set_title('Right Thumb-Wrist')
axs[0, 4].legend()
dist_open = calDist(hand_right_open, wrist_right_open)
dist_close = calDist(hand_right_close, wrist_right_close)
sns.kdeplot(dist_open, ax=axs[0, 5], label='open')
sns.kdeplot(dist_close, ax=axs[0, 5], label='close')
axs[0, 5].set_title('Right Hand-Wrist')
axs[0, 5].legend()
dist_open = calDist(handtip_left_open, thumb_left_open)
dist_close = calDist(handtip_left_close, thumb_left_close)
sns.kdeplot(dist_open, ax=axs[1, 0], label='open')
sns.kdeplot(dist_close, ax=axs[1, 0], label='close')
axs[1, 0].set_title('Left HandTip-Thumb')
axs[1, 0].legend()
dist_open = calDist(handtip_left_open, hand_left_open)
dist_close = calDist(handtip_left_close, hand_left_close)
sns.kdeplot(dist_open, ax=axs[1, 1], label='open')
sns.kdeplot(dist_close, ax=axs[1, 1], label='close')
axs[1, 1].set_title('Left HandTip-Hand')
axs[1, 1].legend()
dist_open = calDist(handtip_left_open, wrist_left_open)
dist_close = calDist(handtip_left_close, wrist_left_close)
sns.kdeplot(dist_open, ax=axs[1, 2], label='open')
sns.kdeplot(dist_close, ax=axs[1, 2], label='close')
axs[1, 2].set_title('Left HandTip-Wrist')
axs[1, 2].legend()
dist_open = calDist(thumb_left_open, hand_left_open)
dist_close = calDist(thumb_left_close, hand_left_close)
sns.kdeplot(dist_open, ax=axs[1, 3], label='open')
sns.kdeplot(dist_close, ax=axs[1, 3], label='close')
axs[1, 3].set_title('Left Thumb-Hand')
axs[1, 3].legend()
dist_open = calDist(thumb_left_open, wrist_left_open)
dist_close = calDist(thumb_left_close, wrist_left_close)
sns.kdeplot(dist_open, ax=axs[1, 4], label='open')
sns.kdeplot(dist_close, ax=axs[1, 4], label='close')
axs[1, 4].set_title('Left Thumb-Wrist')
axs[1, 4].legend()
dist_open = calDist(hand_left_open, wrist_left_open)
dist_close = calDist(hand_left_close, wrist_left_close)
sns.kdeplot(dist_open, ax=axs[1, 5], label='open')
sns.kdeplot(dist_close, ax=axs[1, 5], label='close')
axs[1, 5].set_title('Left Hand-Wrist')
axs[1, 5].legend()
plt.show()


def calDistSplit(data1, data2):
    dist0 = []
    dist_val = []
    for i in range(len(data1)):
        ci1, x1, y1, z1 = data1[i]
        ci2, x2, y2, z2 = data2[i]
        xp = ((x1 - x2) ** 2.0)
        yp = ((y1 - y2) ** 2.0)
        zp = ((z1 - z2) ** 2.0)
        if ci1 == 0 or ci2 == 0:
            dist0.append((xp + yp + zp) ** 0.5)
        else:
            dist_val.append((xp + yp + zp) ** 0.5)
    return dist0, dist_val


# 查看Thumb-Wrist距离By CI
fig, axs = plt.subplots(2, 3)
fig.set_size_inches(8, 5)
dist_open0, dist_open_val = calDistSplit(thumb_right_open, wrist_right_open)
dist_open = dist_open0.copy()
dist_open.extend(dist_open_val)
dist_close0, dist_close_val = calDistSplit(thumb_right_close, wrist_right_close)
dist_close = dist_close0.copy()
dist_close.extend(dist_close_val)
sns.kdeplot(dist_open, ax=axs[0, 0], label='open')
sns.kdeplot(dist_close, ax=axs[0, 0], label='close')
axs[0, 0].set_title('Right Thumb-Wrist-Total')
sns.kdeplot(dist_open0, ax=axs[0, 1], label='open')
sns.kdeplot(dist_close0, ax=axs[0, 1], label='close')
axs[0, 1].set_title('Right Thumb-Wrist-CI=0')
sns.kdeplot(dist_open_val, ax=axs[0, 2], label='open')
sns.kdeplot(dist_close_val, ax=axs[0, 2], label='close')
axs[0, 2].set_title('Right Thumb-Wrist-CI>0')
dist_open0, dist_open_val = calDistSplit(thumb_left_open, wrist_left_open)
dist_open = dist_open0.copy()
dist_open.extend(dist_open_val)
dist_close0, dist_close_val = calDistSplit(thumb_left_close, wrist_left_close)
dist_close = dist_close0.copy()
dist_close.extend(dist_close_val)
sns.kdeplot(dist_open, ax=axs[1, 0], label='open')
sns.kdeplot(dist_close, ax=axs[1, 0], label='close')
axs[1, 0].set_title('Left Thumb-Wrist-Total')
sns.kdeplot(dist_open0, ax=axs[1, 1], label='open')
sns.kdeplot(dist_close0, ax=axs[1, 1], label='close')
axs[1, 1].set_title('Left Thumb-Wrist-CI=0')
sns.kdeplot(dist_open_val, ax=axs[1, 2], label='open')
sns.kdeplot(dist_close_val, ax=axs[1, 2], label='close')
axs[1, 2].set_title('Left Thumb-Wrist-CI>0')
plt.show()

# 查看HandTip-Thumb距离By CI
fig, axs = plt.subplots(2, 3)
fig.set_size_inches(8, 5)
dist_open0, dist_open_val = calDistSplit(thumb_right_open, handtip_right_open)
dist_open = dist_open0.copy()
dist_open.extend(dist_open_val)
dist_close0, dist_close_val = calDistSplit(thumb_right_close, handtip_right_close)
dist_close = dist_close0.copy()
dist_close.extend(dist_close_val)
sns.kdeplot(dist_open, ax=axs[0, 0], label='open')
sns.kdeplot(dist_close, ax=axs[0, 0], label='close')
axs[0, 0].set_title('Right HandTip-Thumb-Total')
sns.kdeplot(dist_open0, ax=axs[0, 1], label='open')
sns.kdeplot(dist_close0, ax=axs[0, 1], label='close')
axs[0, 1].set_title('Right HandTip-Thumb-CI=0')
sns.kdeplot(dist_open_val, ax=axs[0, 2], label='open')
sns.kdeplot(dist_close_val, ax=axs[0, 2], label='close')
axs[0, 2].set_title('Right HandTip-Thumb-CI>0')
dist_open0, dist_open_val = calDistSplit(thumb_left_open, handtip_left_open)
dist_open = dist_open0.copy()
dist_open.extend(dist_open_val)
dist_close0, dist_close_val = calDistSplit(thumb_left_close, handtip_left_close)
dist_close = dist_close0.copy()
dist_close.extend(dist_close_val)
sns.kdeplot(dist_open, ax=axs[1, 0], label='open')
sns.kdeplot(dist_close, ax=axs[1, 0], label='close')
axs[1, 0].set_title('Left HandTip-Thumb-Total')
sns.kdeplot(dist_open0, ax=axs[1, 1], label='open')
sns.kdeplot(dist_close0, ax=axs[1, 1], label='close')
axs[1, 1].set_title('Left HandTip-Thumb-CI=0')
sns.kdeplot(dist_open_val, ax=axs[1, 2], label='open')
sns.kdeplot(dist_close_val, ax=axs[1, 2], label='close')
axs[1, 2].set_title('Left HandTip-Thumb-CI>0')
plt.show()
