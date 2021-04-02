import matplotlib.pyplot as plt
import seaborn as sns


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


def calArea(p1, p2, p3):
    area = []
    for i in range(len(p1)):
        ci1, x1, y1, z1 = p1[i]
        ci2, x2, y2, z2 = p2[i]
        ci3, x3, y3, z3 = p3[i]
        x_v1, y_v1, z_v1 = (x2 - x1), (y2 - y1), (z2 - z1)
        x_v2, y_v2, z_v2 = (x3 - x1), (y3 - y1), (z3 - z1)
        dot_x = (y_v1 * z_v2) - (z_v1 * y_v2)
        dot_y = (z_v1 * x_v2) - (x_v1 * z_v2)
        dot_z = (x_v1 * y_v2) - (y_v1 * x_v2)
        area.append((dot_x ** 2.0 + dot_y ** 2.0 + dot_z ** 2.0) ** 0.5)
    return area


# 查看分布图
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(10, 10)
area_whtt_right_open = calArea(wrist_right_open, thumb_right_open, handtip_right_open)
area_whtt_right_close = calArea(wrist_right_close, thumb_right_close, handtip_right_close)
sns.kdeplot(area_whtt_right_open, ax=axs[0, 0], label='open')
sns.kdeplot(area_whtt_right_close, ax=axs[0, 0], label='close')
axs[0, 0].set_title('Right Wrist-HandTip-Thumb Area')
axs[0, 0].legend()
area_wht_right_open = calArea(wrist_right_open, thumb_right_open, hand_right_open)
area_wht_right_close = calArea(wrist_right_close, thumb_right_close, hand_right_close)
sns.kdeplot(area_wht_right_open, ax=axs[0, 1], label='open')
sns.kdeplot(area_wht_right_close, ax=axs[0, 1], label='close')
axs[0, 1].set_title('Right Wrist-Hand-Thumb Area')
axs[0, 1].legend()
area_whtt_left_open = calArea(wrist_left_open, thumb_left_open, handtip_left_open)
area_whtt_left_close = calArea(wrist_left_close, thumb_left_close, handtip_left_close)
sns.kdeplot(area_whtt_left_open, ax=axs[1, 0], label='open')
sns.kdeplot(area_whtt_left_close, ax=axs[1, 0], label='close')
axs[1, 0].set_title('Left Wrist-HandTip-Thumb Area')
axs[1, 0].legend()
area_wht_left_open = calArea(wrist_left_open, thumb_left_open, hand_left_open)
area_wht_left_close = calArea(wrist_left_close, thumb_left_close, hand_left_close)
sns.kdeplot(area_wht_left_open, ax=axs[1, 1], label='open')
sns.kdeplot(area_wht_left_close, ax=axs[1, 1], label='close')
axs[1, 1].set_title('Left Wrist-Hand-Thumb Area')
axs[1, 1].legend()
plt.show()


