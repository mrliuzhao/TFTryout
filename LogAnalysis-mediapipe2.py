import matplotlib.pyplot as plt
import seaborn as sns

def extractFileInfo(logfile):
    hand_right = []
    hand_left = []
    handtip_right = []
    handtip_left = []
    thumb_right = []
    thumb_left = []
    wrist_right = []
    wrist_left = []
    failCount = 0
    with open(logfile) as f:
        for line in f:
            if 'Fail to detect all landmarks' in line:
                failCount += 1
            else:
                start = line.index('Position:(') + len('Position:(')
                temp = line[start:-2].split(',')
                if 'LeftWrist' in line:
                    wrist_left.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'RightWrist' in line:
                    wrist_right.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'LeftThumb' in line:
                    thumb_left.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'LeftPalm' in line:
                    hand_left.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'LeftTip' in line:
                    handtip_left.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'RightThumb' in line:
                    thumb_right.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'RightPalm' in line:
                    hand_right.append((float(temp[0]), float(temp[1]), float(temp[2])))
                elif 'RightTip' in line:
                    handtip_right.append((float(temp[0]), float(temp[1]), float(temp[2])))

    return failCount, hand_right, hand_left, handtip_right, handtip_left, thumb_right, thumb_left, wrist_right, wrist_left


filename = r'C:\Users\Administrator\Desktop\hand-open-raw.log'
failCount_open, hand_right_open, hand_left_open, handtip_right_open, handtip_left_open, thumb_right_open, thumb_left_open, wrist_right_open, wrist_left_open = extractFileInfo(filename)

filename = r'C:\Users\Administrator\Desktop\hand-close-raw.log'
failCount_close, hand_right_close, hand_left_close, handtip_right_close, handtip_left_close, thumb_right_close, thumb_left_close, wrist_right_close, wrist_left_close = extractFileInfo(filename)

filename = r'C:\Users\Administrator\Desktop\hand-grasp-raw.log'
failCount_grasp, hand_right_grasp, hand_left_grasp, handtip_right_grasp, handtip_left_grasp, thumb_right_grasp, thumb_left_grasp, wrist_right_grasp, wrist_left_grasp = extractFileInfo(filename)


def calDist(data1, data2):
    dist = []
    for i in range(len(data1)):
        x1, y1, z1 = data1[i]
        x2, y2, z2 = data2[i]
        xp = ((x1 - x2) ** 2.0)
        yp = ((y1 - y2) ** 2.0)
        zp = ((z1 - z2) ** 2.0)
        dist.append((xp + yp + zp) ** 0.5)
    return dist


# 绘制箱线图
fig, axs = plt.subplots(2, 6)
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

# 查看各种距离分布图
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










