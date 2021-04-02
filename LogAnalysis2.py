import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# filename = r'C:\Users\Administrator\Desktop\logs(77)\kinect-2020-12-03.log'
# filename = r'C:\Users\Administrator\Desktop\logs(78)\kinect-2020-12-03.log'
# filename = r'C:\Users\Administrator\Desktop\手部识别\HandOpen.log'
filename = r'C:\Users\Administrator\Desktop\手部识别\HandClose2.log'

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
with open(filename) as f:
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


# 检查CI
def checkCI(data):
    ci0_idx = []
    ci1_idx = []
    ci2_idx = []
    for idx, (ci, x, y, z) in enumerate(data):
        if ci == 0:
            ci0_idx.append(idx)
        elif ci == 1:
            ci1_idx.append(idx)
        elif ci == 2:
            ci2_idx.append(idx)
        else:
            print('Invalid CI values!!! ci: ' + str(ci))
    return ci0_idx, ci1_idx, ci2_idx


ci0_idx, ci1_idx, ci2_idx = checkCI(shoulder_right)


def splitData(data):
    cis = []
    pos_x = []
    pos_y = []
    pos_z = []
    for idx, (ci, x, y, z) in enumerate(data):
        cis.append(ci)
        pos_x.append(x)
        pos_y.append(y)
        pos_z.append(z)
    return cis, pos_x, pos_y, pos_z


fig, axs = plt.subplots(3, 2)
cis, pos_x, pos_y, pos_z = splitData(hand_right)
axs[0, 0].plot(cis)
axs[0, 0].set_title('Hand Right CIs')
cis, pos_x, pos_y, pos_z = splitData(hand_left)
axs[0, 1].plot(cis)
axs[0, 1].set_title('Hand Left CIs')
cis, pos_x, pos_y, pos_z = splitData(handtip_right)
axs[1, 0].plot(cis)
axs[1, 0].set_title('Hand Tip Right CIs')
cis, pos_x, pos_y, pos_z = splitData(handtip_left)
axs[1, 1].plot(cis)
axs[1, 1].set_title('Hand Tip Left CIs')
cis, pos_x, pos_y, pos_z = splitData(thumb_right)
axs[2, 0].plot(cis)
axs[2, 0].set_title('Thumb Right CIs')
cis, pos_x, pos_y, pos_z = splitData(thumb_left)
axs[2, 1].plot(cis)
axs[2, 1].set_title('Thumb Left CIs')
plt.show()


cis, pos_x, pos_y, pos_z = splitData(hand_right)
# 双纵轴比较
fig, ax1 = plt.subplots()
ax1.plot(cis, color='blue', label="ci")
ax2 = ax1.twinx()
ax2.plot(pos_x, color='red', label="position x")
ax2.plot(pos_y, color='orange', label="position y")
ax2.plot(pos_z, color='green', label="position z")
plt.show()


def splitPosByCI(data):
    pos0_x = []
    pos0_y = []
    pos0_z = []
    pos1_x = []
    pos1_y = []
    pos1_z = []
    pos2_x = []
    pos2_y = []
    pos2_z = []
    for idx, (ci, x, y, z) in enumerate(data):
        if ci == 0:
            pos0_x.append(x)
            pos0_y.append(y)
            pos0_z.append(z)
        elif ci == 1:
            pos1_x.append(x)
            pos1_y.append(y)
            pos1_z.append(z)
        elif ci == 2:
            pos2_x.append(x)
            pos2_y.append(y)
            pos2_z.append(z)
        else:
            print('invalid ci value! ci: ' + str(ci))
    return pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z


fig, axs = plt.subplots(3, 6, sharex='col', sharey='row')
fig.set_size_inches(16, 9)
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(hand_right)
axs[0, 0].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 0].set_title('Hand Right-x')
axs[1, 0].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 0].set_title('Hand Right-y')
axs[2, 0].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 0].set_title('Hand Right-z')
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(handtip_right)
axs[0, 1].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 1].set_title('Hand Tip Right-x')
axs[1, 1].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 1].set_title('Hand Tip Right-y')
axs[2, 1].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 1].set_title('Hand Tip Right-z')
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(thumb_right)
axs[0, 2].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 2].set_title('Thumb Right-x')
axs[1, 2].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 2].set_title('Thumb Right-y')
axs[2, 2].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 2].set_title('Thumb Right-z')
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(hand_left)
axs[0, 3].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 3].set_title('Hand Left-x')
axs[1, 3].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 3].set_title('Hand Left-y')
axs[2, 3].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 3].set_title('Hand Left-z')
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(handtip_left)
axs[0, 4].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 4].set_title('Hand Tip Left-x')
axs[1, 4].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 4].set_title('Hand Tip Left-y')
axs[2, 4].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 4].set_title('Hand Tip Left-z')
pos0_x, pos0_y, pos0_z, pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z = splitPosByCI(thumb_left)
axs[0, 5].boxplot([pos0_x, pos1_x, pos2_x])
axs[0, 5].set_title('Thumb Left-x')
axs[1, 5].boxplot([pos0_y, pos1_y, pos2_y])
axs[1, 5].set_title('Thumb Left-y')
axs[2, 5].boxplot([pos0_z, pos1_z, pos2_z])
axs[2, 5].set_title('Thumb Left-z')
plt.show()

# 3D展示散点图
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(pos0_x, pos0_y, pos0_z, color='green', label='ci=0')
# ax.scatter(pos1_x, pos1_y, pos1_z, color='orange', label='ci=1')
# ax.scatter(pos2_x, pos2_y, pos2_z, color='red', label='ci=2')
# ax.legend(loc='best')
# ax.set_zlabel('Z', fontdict={'size': 15})
# ax.set_ylabel('Y', fontdict={'size': 15})
# ax.set_xlabel('X', fontdict={'size': 15})
#
# plt.title('Right Thumb Position')
# plt.show()


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


fig, axs = plt.subplots(2, 6, sharex='col', sharey='row')
fig.set_size_inches(15, 5)
dist0, dist_val = calDist(handtip_right, thumb_right)
axs[0, 0].boxplot([dist0, dist_val])
axs[0, 0].set_title('Right HandTip-Thumb')
axs[0, 0].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(handtip_right, hand_right)
axs[0, 1].boxplot([dist0, dist_val])
axs[0, 1].set_title('Right HandTip-Hand')
axs[0, 1].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(handtip_right, wrist_right)
axs[0, 2].boxplot([dist0, dist_val])
axs[0, 2].set_title('Right HandTip-Wrist')
axs[0, 2].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(thumb_right, hand_right)
axs[0, 3].boxplot([dist0, dist_val])
axs[0, 3].set_title('Right Thumb-Hand')
axs[0, 3].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(thumb_right, wrist_right)
axs[0, 4].boxplot([dist0, dist_val])
axs[0, 4].set_title('Right Thumb-Wrist')
axs[0, 4].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(hand_right, wrist_right)
axs[0, 5].boxplot([dist0, dist_val])
axs[0, 5].set_title('Right Hand-Wrist')
axs[0, 5].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(handtip_left, thumb_left)
axs[1, 0].boxplot([dist0, dist_val])
axs[1, 0].set_title('Left HandTip-Thumb')
axs[1, 0].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(handtip_left, hand_left)
axs[1, 1].boxplot([dist0, dist_val])
axs[1, 1].set_title('Left HandTip-Hand')
axs[1, 1].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(handtip_left, wrist_left)
axs[1, 2].boxplot([dist0, dist_val])
axs[1, 2].set_title('Left HandTip-Wrist')
axs[1, 2].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(thumb_left, hand_left)
axs[1, 3].boxplot([dist0, dist_val])
axs[1, 3].set_title('Left Thumb-Hand')
axs[1, 3].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(thumb_left, wrist_left)
axs[1, 4].boxplot([dist0, dist_val])
axs[1, 4].set_title('Left Thumb-Wrist')
axs[1, 4].set_xticklabels(['CI=0', 'CI>0'])
dist0, dist_val = calDist(hand_left, wrist_left)
axs[1, 5].boxplot([dist0, dist_val])
axs[1, 5].set_title('Left Hand-Wrist')
axs[1, 5].set_xticklabels(['CI=0', 'CI>0'])

plt.show()










