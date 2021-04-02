import matplotlib.pyplot as plt

# filename = r'C:\Users\Administrator\Desktop\hand-open-WFOV-conv.log'
# filename = r'C:\Users\Administrator\Desktop\hand-close-WFOV-conv.log'
# filename = r'C:\Users\Administrator\Desktop\hand-grasp-WFOV-conv.log'

# filename = r'D:\GDCC\docs\技术文档\MediaPipe手部识别\open-hand-NFOV-1080p.log'
# filename = r'C:\Users\Administrator\Desktop\hand-open-raw.log'
# filename = r'C:\Users\Administrator\Desktop\hand-close-raw.log'
filename = r'C:\Users\Administrator\Desktop\hand-grasp-raw.log'

hand_right = []
hand_left = []
handtip_right = []
handtip_left = []
thumb_right = []
thumb_left = []
wrist_right = []
wrist_left = []
mediapipeFailCount = 0
with open(filename) as f:
    for line in f:
        if 'Fail to detect all landmarks' in line:
            mediapipeFailCount += 1
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

rate = 100.0 * mediapipeFailCount / (len(hand_right) + mediapipeFailCount)


def splitData(data):
    pos_x = []
    pos_y = []
    pos_z = []
    z0count = 0
    zovercount = 0
    for idx, (x, y, z) in enumerate(data):
        if z == 0:
            z0count += 1
        if z > 3000:
            zovercount += 1
        pos_x.append(x)
        pos_y.append(y)
        pos_z.append(z)
    return pos_x, pos_y, pos_z, z0count, zovercount


print('----- Invalid Z value Statistics -----')
# 检查Z轴数据
fig, axs = plt.subplots(4, 2)
pos_x, pos_y, pos_z, z0count, zovercount = splitData(wrist_right)
axs[0, 0].plot(pos_z)
axs[0, 0].set_title('Wrist Right Z value')
print('z=0: Wrist Right: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Wrist Right: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(wrist_left)
axs[0, 1].plot(pos_z)
axs[0, 1].set_title('Wrist Left Z value')
print('z=0: Wrist Left: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Wrist Left: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(hand_right)
axs[1, 0].plot(pos_z)
axs[1, 0].set_title('Hand Right Z value')
print('z=0: Hand Right: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Hand Right: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(hand_left)
axs[1, 1].plot(pos_z)
axs[1, 1].set_title('Hand Left Z value')
print('z=0: Hand Left: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Hand Left: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(handtip_right)
axs[2, 0].plot(pos_z)
axs[2, 0].set_title('Hand Tip Right Z value')
print('z=0: Hand Tip Right: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Hand Tip Right: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(handtip_left)
axs[2, 1].plot(pos_z)
axs[2, 1].set_title('Hand Tip Left Z value')
print('z=0: Hand Tip Left: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Hand Tip Left: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(thumb_right)
axs[3, 0].plot(pos_z)
axs[3, 0].set_title('Thumb Right Z value')
print('z=0: Thumb Right: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Thumb Right: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
pos_x, pos_y, pos_z, z0count, zovercount = splitData(thumb_left)
axs[3, 1].plot(pos_z)
axs[3, 1].set_title('Thumb Left Z value')
print('z=0: Thumb Left: ' + str(z0count) + ' - ' + str(100.0 * z0count / len(wrist_right)))
print('z>3000: Thumb Left: ' + str(zovercount) + ' - ' + str(100.0 * zovercount / len(wrist_right)))
plt.show()

# 过滤所有0数据
validCount = 0
hand_right_filtered = []
hand_left_filtered = []
handtip_right_filtered = []
handtip_left_filtered = []
thumb_right_filtered = []
thumb_left_filtered = []
wrist_right_filtered = []
wrist_left_filtered = []
for i in range(len(hand_right)):
    x, y, z = hand_right[i]
    if z == 0:
        continue
    x, y, z = hand_left[i]
    if z == 0:
        continue
    x, y, z = handtip_right[i]
    if z == 0:
        continue
    x, y, z = handtip_left[i]
    if z == 0:
        continue
    x, y, z = thumb_right[i]
    if z == 0:
        continue
    x, y, z = thumb_left[i]
    if z == 0:
        continue
    x, y, z = wrist_right[i]
    if z == 0:
        continue
    x, y, z = wrist_left[i]
    if z == 0:
        continue
    validCount += 1
    hand_right_filtered.append(hand_right[i])
    hand_left_filtered.append(hand_left[i])
    handtip_right_filtered.append(handtip_right[i])
    handtip_left_filtered.append(handtip_left[i])
    thumb_right_filtered.append(thumb_right[i])
    thumb_left_filtered.append(thumb_left[i])
    wrist_right_filtered.append(wrist_right[i])
    wrist_left_filtered.append(wrist_left[i])

print('Valid Count: ' + str(validCount) + ' - ' + str(100.0 * validCount / len(wrist_right)))
print('综合成功率: ' + str(validCount) + ' - ' + str(100.0 * validCount / 1800))

# 检查Z轴过滤数据
fig, axs = plt.subplots(4, 2)
pos_x, pos_y, pos_z, z0count, zovercount = splitData(wrist_right_filtered)
axs[0, 0].plot(pos_z)
axs[0, 0].set_title('Wrist Right Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(wrist_left_filtered)
axs[0, 1].plot(pos_z)
axs[0, 1].set_title('Wrist Left Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(hand_right_filtered)
axs[1, 0].plot(pos_z)
axs[1, 0].set_title('Hand Right Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(hand_left_filtered)
axs[1, 1].plot(pos_z)
axs[1, 1].set_title('Hand Left Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(handtip_right_filtered)
axs[2, 0].plot(pos_z)
axs[2, 0].set_title('Hand Tip Right Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(handtip_left_filtered)
axs[2, 1].plot(pos_z)
axs[2, 1].set_title('Hand Tip Left Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(thumb_right_filtered)
axs[3, 0].plot(pos_z)
axs[3, 0].set_title('Thumb Right Z value')
pos_x, pos_y, pos_z, z0count, zovercount = splitData(thumb_left_filtered)
axs[3, 1].plot(pos_z)
axs[3, 1].set_title('Thumb Left Z value')
plt.show()


# 再过滤所有超过3000的数据
validCount = 0
hand_right_filtered2 = []
hand_left_filtered2 = []
handtip_right_filtered2 = []
handtip_left_filtered2 = []
thumb_right_filtered2 = []
thumb_left_filtered2 = []
wrist_right_filtered2 = []
wrist_left_filtered2 = []
for i in range(len(hand_right)):
    x, y, z = hand_right[i]
    if z == 0:
        continue
    x, y, z = hand_left[i]
    if z == 0:
        continue
    x, y, z = handtip_right[i]
    if z == 0:
        continue
    x, y, z = handtip_left[i]
    if z == 0:
        continue
    x, y, z = thumb_right[i]
    if z == 0:
        continue
    x, y, z = thumb_left[i]
    if z == 0:
        continue
    x, y, z = wrist_right[i]
    if z == 0:
        continue
    x, y, z = wrist_left[i]
    if z == 0:
        continue
    validCount += 1
    hand_right_filtered.append(hand_right[i])
    hand_left_filtered.append(hand_left[i])
    handtip_right_filtered.append(handtip_right[i])
    handtip_left_filtered.append(handtip_left[i])
    thumb_right_filtered.append(thumb_right[i])
    thumb_left_filtered.append(thumb_left[i])
    wrist_right_filtered.append(wrist_right[i])
    wrist_left_filtered.append(wrist_left[i])











