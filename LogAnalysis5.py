import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# filename = r'C:\Users\Administrator\Desktop\HandOpen.log'
# filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandGrasp.log'
# filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandClose.log'
# filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandOpen.log'

def analyseFile(logfile):
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


    def calDist(data1, data2):
        dist0 = []
        dist_valid = []
        for i in range(len(data1)):
            ci1, x1, y1, z1 = data1[i]
            ci2, x2, y2, z2 = data2[i]
            xp = ((x1 - x2) ** 2.0)
            yp = ((y1 - y2) ** 2.0)
            zp = ((z1 - z2) ** 2.0)
            if ci1 == 0 or ci2 == 0:
                dist0.append((xp + yp + zp) ** 0.5)
            else:
                dist_valid.append((xp + yp + zp) ** 0.5)
        return dist0, dist_valid


    # fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
    # fig.set_size_inches(15, 5)
    # dist0, dist_val = calDist(handtip_right, thumb_right)
    # axs[0, 0].boxplot([dist0, dist_val])
    # axs[0, 0].set_title('Right HandTip-Thumb')
    # axs[0, 0].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(handtip_right, hand_right)
    # axs[0, 1].boxplot([dist0, dist_val])
    # axs[0, 1].set_title('Right HandTip-Hand')
    # axs[0, 1].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(handtip_right, wrist_right)
    # axs[0, 2].boxplot([dist0, dist_val])
    # axs[0, 2].set_title('Right HandTip-Wrist')
    # axs[0, 2].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(thumb_right, hand_right)
    # axs[0, 3].boxplot([dist0, dist_val])
    # axs[0, 3].set_title('Right Thumb-Hand')
    # axs[0, 3].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(thumb_right, wrist_right)
    # axs[0, 4].boxplot([dist0, dist_val])
    # axs[0, 4].set_title('Right Thumb-Wrist')
    # axs[0, 4].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(hand_right, wrist_right)
    # axs[0, 5].boxplot([dist0, dist_val])
    # axs[0, 5].set_title('Right Hand-Wrist')
    # axs[0, 5].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(handtip_left, thumb_left)
    # axs[1, 0].boxplot([dist0, dist_val])
    # axs[1, 0].set_title('Left HandTip-Thumb')
    # axs[1, 0].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(handtip_left, hand_left)
    # axs[1, 1].boxplot([dist0, dist_val])
    # axs[1, 1].set_title('Left HandTip-Hand')
    # axs[1, 1].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(handtip_left, wrist_left)
    # axs[1, 2].boxplot([dist0, dist_val])
    # axs[1, 2].set_title('Left HandTip-Wrist')
    # axs[1, 2].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(thumb_left, hand_left)
    # axs[1, 3].boxplot([dist0, dist_val])
    # axs[1, 3].set_title('Left Thumb-Hand')
    # axs[1, 3].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(thumb_left, wrist_left)
    # axs[1, 4].boxplot([dist0, dist_val])
    # axs[1, 4].set_title('Left Thumb-Wrist')
    # axs[1, 4].set_xticklabels(['CI=0', 'CI>0'])
    # dist0, dist_val = calDist(hand_left, wrist_left)
    # axs[1, 5].boxplot([dist0, dist_val])
    # axs[1, 5].set_title('Left Hand-Wrist')
    # axs[1, 5].set_xticklabels(['CI=0', 'CI>0'])
    # plt.show()


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


    # 查看折线图
    import random
    length = 30
    start = random.randint(0, 1800 - length)
    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
    fig.set_size_inches(15, 5)
    dist = calDist(handtip_right[start:start+length], thumb_right[start:start+length])
    axs[0, 0].plot(dist)
    axs[0, 0].set_title('Right HandTip-Thumb')
    dist = calDist(handtip_right[start:start+length], hand_right[start:start+length])
    axs[0, 1].plot(dist)
    axs[0, 1].set_title('Right HandTip-Hand')
    dist = calDist(handtip_right[start:start+length], wrist_right[start:start+length])
    axs[0, 2].plot(dist)
    axs[0, 2].set_title('Right HandTip-Wrist')
    dist = calDist(thumb_right[start:start+length], hand_right[start:start+length])
    axs[0, 3].plot(dist)
    axs[0, 3].set_title('Right Thumb-Hand')
    dist = calDist(thumb_right[start:start+length], wrist_right[start:start+length])
    axs[0, 4].plot(dist)
    axs[0, 4].set_title('Right Thumb-Wrist')
    dist = calDist(hand_right[start:start+length], wrist_right[start:start+length])
    axs[0, 5].plot(dist)
    axs[0, 5].set_title('Right Hand-Wrist')
    dist = calDist(handtip_left[start:start+length], thumb_left[start:start+length])
    axs[1, 0].plot(dist)
    axs[1, 0].set_title('Left HandTip-Thumb')
    dist = calDist(handtip_left[start:start+length], hand_left[start:start+length])
    axs[1, 1].plot(dist)
    axs[1, 1].set_title('Left HandTip-Hand')
    dist = calDist(handtip_left[start:start+length], wrist_left[start:start+length])
    axs[1, 2].plot(dist)
    axs[1, 2].set_title('Left HandTip-Wrist')
    dist = calDist(thumb_left[start:start+length], hand_left[start:start+length])
    axs[1, 3].plot(dist)
    axs[1, 3].set_title('Left Thumb-Hand')
    dist = calDist(thumb_left[start:start+length], wrist_left[start:start+length])
    axs[1, 4].plot(dist)
    axs[1, 4].set_title('Left Thumb-Wrist')
    dist = calDist(hand_left[start:start+length], wrist_left[start:start+length])
    axs[1, 5].plot(dist)
    axs[1, 5].set_title('Left Hand-Wrist')
    plt.show()


filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandClose.log'
analyseFile(filename)
filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandOpen.log'
analyseFile(filename)
filename = r'D:\GDCC\docs\技术文档\AzureKinect手部识别\HandGrasp.log'
analyseFile(filename)



# def calDeltaDist(dists):
#     delta_dist = []
#     for i in range(len(dists) - 1):
#         delta_dist.append(abs(dists[i+1] - dists[i]))
#     return delta_dist
#
#
# # 查看距离差值图
# fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
# fig.set_size_inches(15, 5)
# dist = calDist(handtip_right, thumb_right)
# axs[0, 0].plot(calDeltaDist(dist))
# axs[0, 0].set_title('Right HandTip-Thumb')
# dist = calDist(handtip_right, hand_right)
# axs[0, 1].plot(calDeltaDist(dist))
# axs[0, 1].set_title('Right HandTip-Hand')
# dist = calDist(handtip_right, wrist_right)
# axs[0, 2].plot(calDeltaDist(dist))
# axs[0, 2].set_title('Right HandTip-Wrist')
# dist = calDist(thumb_right, hand_right)
# axs[0, 3].plot(calDeltaDist(dist))
# axs[0, 3].set_title('Right Thumb-Hand')
# dist = calDist(thumb_right, wrist_right)
# axs[0, 4].plot(calDeltaDist(dist))
# axs[0, 4].set_title('Right Thumb-Wrist')
# dist = calDist(hand_right, wrist_right)
# axs[0, 5].plot(calDeltaDist(dist))
# axs[0, 5].set_title('Right Hand-Wrist')
# dist = calDist(handtip_left, thumb_left)
# axs[1, 0].plot(calDeltaDist(dist))
# axs[1, 0].set_title('Left HandTip-Thumb')
# dist = calDist(handtip_left, hand_left)
# axs[1, 1].plot(calDeltaDist(dist))
# axs[1, 1].set_title('Left HandTip-Hand')
# dist = calDist(handtip_left, wrist_left)
# axs[1, 2].plot(calDeltaDist(dist))
# axs[1, 2].set_title('Left HandTip-Wrist')
# dist = calDist(thumb_left, hand_left)
# axs[1, 3].plot(calDeltaDist(dist))
# axs[1, 3].set_title('Left Thumb-Hand')
# dist = calDist(thumb_left, wrist_left)
# axs[1, 4].plot(calDeltaDist(dist))
# axs[1, 4].set_title('Left Thumb-Wrist')
# dist = calDist(hand_left, wrist_left)
# axs[1, 5].plot(calDeltaDist(dist))
# axs[1, 5].set_title('Left Hand-Wrist')
# plt.show()
#
# # 查看距离差值箱线图
# fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
# fig.set_size_inches(15, 5)
# dist = calDist(handtip_right, thumb_right)
# deltas = calDeltaDist(dist)
# axs[0, 0].boxplot(deltas)
# axs[0, 0].set_title('Right HandTip-Thumb')
# dist = calDist(handtip_right, hand_right)
# deltas = calDeltaDist(dist)
# axs[0, 1].boxplot(deltas)
# axs[0, 1].set_title('Right HandTip-Hand')
# dist = calDist(handtip_right, wrist_right)
# deltas = calDeltaDist(dist)
# axs[0, 2].boxplot(deltas)
# axs[0, 2].set_title('Right HandTip-Wrist')
# dist = calDist(thumb_right, hand_right)
# deltas = calDeltaDist(dist)
# axs[0, 3].boxplot(deltas)
# axs[0, 3].set_title('Right Thumb-Hand')
# dist = calDist(thumb_right, wrist_right)
# deltas = calDeltaDist(dist)
# axs[0, 4].boxplot(deltas)
# axs[0, 4].set_title('Right Thumb-Wrist')
# dist = calDist(hand_right, wrist_right)
# deltas = calDeltaDist(dist)
# axs[0, 5].boxplot(deltas)
# axs[0, 5].set_title('Right Hand-Wrist')
# dist = calDist(handtip_left, thumb_left)
# deltas = calDeltaDist(dist)
# axs[1, 0].boxplot(deltas)
# axs[1, 0].set_title('Left HandTip-Thumb')
# dist = calDist(handtip_left, hand_left)
# deltas = calDeltaDist(dist)
# axs[1, 1].boxplot(deltas)
# axs[1, 1].set_title('Left HandTip-Hand')
# dist = calDist(handtip_left, wrist_left)
# deltas = calDeltaDist(dist)
# axs[1, 2].boxplot(deltas)
# axs[1, 2].set_title('Left HandTip-Wrist')
# dist = calDist(thumb_left, hand_left)
# deltas = calDeltaDist(dist)
# axs[1, 3].boxplot(deltas)
# axs[1, 3].set_title('Left Thumb-Hand')
# dist = calDist(thumb_left, wrist_left)
# deltas = calDeltaDist(dist)
# axs[1, 4].boxplot(deltas)
# axs[1, 4].set_title('Left Thumb-Wrist')
# dist = calDist(hand_left, wrist_left)
# deltas = calDeltaDist(dist)
# axs[1, 5].boxplot(deltas)
# axs[1, 5].set_title('Left Hand-Wrist')
# plt.show()
#
# # 查看距离差值分布图
# fig, axs = plt.subplots(2, 6, sharex='all')
# fig.set_size_inches(15, 5)
# dist = calDist(handtip_right, thumb_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 0])
# axs[0, 0].set_title('Right HandTip-Thumb')
# dist = calDist(handtip_right, hand_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 1])
# axs[0, 1].set_title('Right HandTip-Hand')
# dist = calDist(handtip_right, wrist_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 2])
# axs[0, 2].set_title('Right HandTip-Wrist')
# dist = calDist(thumb_right, hand_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 3])
# axs[0, 3].set_title('Right Thumb-Hand')
# dist = calDist(thumb_right, wrist_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 4])
# axs[0, 4].set_title('Right Thumb-Wrist')
# dist = calDist(hand_right, wrist_right)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[0, 5])
# axs[0, 5].set_title('Right Hand-Wrist')
# dist = calDist(handtip_left, thumb_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 0])
# axs[1, 0].set_title('Left HandTip-Thumb')
# dist = calDist(handtip_left, hand_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 1])
# axs[1, 1].set_title('Left HandTip-Hand')
# dist = calDist(handtip_left, wrist_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 2])
# axs[1, 2].set_title('Left HandTip-Wrist')
# dist = calDist(thumb_left, hand_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 3])
# axs[1, 3].set_title('Left Thumb-Hand')
# dist = calDist(thumb_left, wrist_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 4])
# axs[1, 4].set_title('Left Thumb-Wrist')
# dist = calDist(hand_left, wrist_left)
# deltas = calDeltaDist(dist)
# sns.kdeplot(deltas, ax=axs[1, 5])
# axs[1, 5].set_title('Left Hand-Wrist')
# plt.show()

# 查看距离差值-对数分布图
# EPSILON = 1e-8
# fig, axs = plt.subplots(2, 6)
# fig.set_size_inches(15, 5)
# dist = calDist(handtip_right, thumb_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 0])
# axs[0, 0].set_title('Right HandTip-Thumb')
# dist = calDist(handtip_right, hand_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 1])
# axs[0, 1].set_title('Right HandTip-Hand')
# dist = calDist(handtip_right, wrist_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 2])
# axs[0, 2].set_title('Right HandTip-Wrist')
# dist = calDist(thumb_right, hand_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 3])
# axs[0, 3].set_title('Right Thumb-Hand')
# dist = calDist(thumb_right, wrist_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 4])
# axs[0, 4].set_title('Right Thumb-Wrist')
# dist = calDist(hand_right, wrist_right)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[0, 5])
# axs[0, 5].set_title('Right Hand-Wrist')
# dist = calDist(handtip_left, thumb_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 0])
# axs[1, 0].set_title('Left HandTip-Thumb')
# dist = calDist(handtip_left, hand_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 1])
# axs[1, 1].set_title('Left HandTip-Hand')
# dist = calDist(handtip_left, wrist_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 2])
# axs[1, 2].set_title('Left HandTip-Wrist')
# dist = calDist(thumb_left, hand_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 3])
# axs[1, 3].set_title('Left Thumb-Hand')
# dist = calDist(thumb_left, wrist_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 4])
# axs[1, 4].set_title('Left Thumb-Wrist')
# dist = calDist(hand_left, wrist_left)
# deltas = calDeltaDist(dist)
# delta_log = np.log(np.asarray(deltas) + EPSILON)
# sns.kdeplot(delta_log, ax=axs[1, 5])
# axs[1, 5].set_title('Left Hand-Wrist')
# plt.show()


