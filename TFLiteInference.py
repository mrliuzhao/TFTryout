import cv2
import tensorflow.lite as tflite
import numpy as np
import time

# val options = Interpreter.Options()
# options.setNumThreads(NUM_LITE_THREADS)
# when(device)
# {
#     Device.CPU -> {}
# Device.GPU -> {
#     gpuDelegate = GpuDelegate()
# options.addDelegate(gpuDelegate)
# }
# Device.NNAPI -> options.setUseNNAPI(true)
# }
# interpreter = Interpreter(loadModelFile(filename, context), options)
# return interpreter!!

# NUM_LITE_THREADS = 4


# 初始化interpreter
# model_file = r'C:\Users\Administrator\Desktop\TFLiteModels\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
model_file = r'C:\Users\Administrator\Desktop\TFLiteModels\multi_personInceptionV3TL.h5enet_v1_075_float.tflite'
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# 获取输入输出详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# NxHxWxC, H:1, W:2
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# 读取视频流
cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 640
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 480

# 比较视频流和模型输入大小的差别
frame_hwfactor = frame_height / frame_width
input_hwfactor = input_height / input_width

# 计算视频流图像需要裁切的大小，将多出部分两边等长裁切
delta_height = 0
delta_width = 0
# 视频流更瘦长，则裁剪高
if frame_hwfactor > input_hwfactor:
    delta_height = (frame_height - (input_hwfactor * frame_width)) / 2.0
elif frame_hwfactor < input_hwfactor: # 视频流更矮胖，则裁剪宽
    delta_width = (frame_width - (frame_height / input_hwfactor)) / 2.0
width_start = int(delta_width)
width_end = int(frame_width - delta_width)
height_start = int(delta_height)
height_end = int(frame_height - delta_height)
scale_factor = input_height / (frame_height - (2 * delta_height))
last_time = None
while True:
    _, frame = cap.read()

    # 裁剪图片
    roi = frame[height_start:height_end, width_start:width_end, :]
    shrinked = cv2.resize(roi, (input_width, input_height), interpolation=cv2.INTER_AREA)

    # add N dim
    input_data = np.expand_dims(shrinked, axis=0)
    # Normalize input to [-1,1]
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # 设置输入
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 调用推断
    interpreter.invoke()

    # 获取输出
    heatMap = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])
    _, block_rows, block_cols, landmark_size = heatMap.shape  # 1, 9, 9, 17

    # 遍历找出各个关节点可能性最大的块
    keypoints_blockpos = []
    for l in range(landmark_size):
        maxP = 0.0
        max_r = 0
        max_c = 0
        for r in range(block_rows):
            for c in range(block_cols):
                if heatMap[0][r][c][l] > maxP:
                    maxP = heatMap[0][r][c][l]
                    max_r = r
                    max_c = c
        keypoints_blockpos.append((max_r, max_c, maxP))

    # 结合offset找到具体像素级别坐标
    keypoints_pos = []
    for l in range(landmark_size):
        base_y = keypoints_blockpos[l][0]
        base_x = keypoints_blockpos[l][1]
        pos_y = int(base_y / (block_rows - 1) * input_height + offsets[0][base_y][base_x][l])
        pos_x = int(base_x / (block_cols - 1) * input_width + offsets[0][base_y][base_x][l + landmark_size])
        keypoints_pos.append((pos_x, pos_y))

    # 画出鼻子
    nose_pos = keypoints_pos[0]
    if nose_pos[0] > 0 or nose_pos[1] > 0:
        cv2.circle(shrinked, (nose_pos[0], nose_pos[1]), 2, color=(0, 0, 255), thickness=cv2.FILLED)
        # 坐标map回原图
        x = int(nose_pos[0] / scale_factor + delta_width)
        y = int(nose_pos[1] / scale_factor + delta_height)
        cv2.circle(frame, (x, y), 2, color=(0, 0, 255), thickness=cv2.FILLED)

    now = time.time()
    if last_time is not None:
        time_elapse = (now - last_time)
        fps = 1.0 / time_elapse
        info = "FPS: "
        cv2.putText(frame, text='FPS: {:.2f}'.format(fps), org=(0, 50), color=(0, 0, 0),
                    thickness=2, fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    last_time = now

    # cv2.imshow('Camera', shrinked)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


