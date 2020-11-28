import cv2
import math
import numpy as np

INPUT_SIZE = 56
BIAS_W = [7, 12, 22]
BIAS_H = [12, 19, 29]

# nms算法
def nms(dets, thresh=0.35):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bbox
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    return keep

# 定义sigmod函数
def sigmod(x):
    return 1.0 / (1.0 + math.exp(-x))

# 处理前向输出feature_map
def feature_map_handle(length, shape, test_img, box_list):
    ih, iw, _ = test_img.shape
    confidence = 0.75
    for i in range(length):
        for j in range(length):
            anchors_boxs_shape = shape[i][j].reshape((3, 6))
            # 将每个预测框向量包含信息迭代出来
            for k in range(3):
                anchors_box = anchors_boxs_shape[k]
                # 计算实际置信度,阀值处理,anchors_box[7]
                score = sigmod(anchors_box[4])
                if score > confidence:
                    # tolist()数组转list
                    cls_list = anchors_box[5:6].tolist()
                    label = cls_list.index(max(cls_list))
                    obj_score = score
                    x = ((sigmod(anchors_box[0]) + i) / float(length)) * iw
                    y = ((sigmod(anchors_box[1]) + j) / float(length)) * ih

                    w = (((BIAS_W[k]) * math.exp(anchors_box[2])) / INPUT_SIZE) * iw
                    h = (((BIAS_H[k]) * math.exp(anchors_box[3])) / INPUT_SIZE) * ih
                    x1 = int(x - w * 0.5)
                    x2 = int(x + w * 0.5)
                    y1 = int(y - h * 0.2)
                    y2 = int(y + h * 0.5)
                    box_list.append([x1, y1, x2, y2, round(obj_score, 4), label])

# 3个feature_map的预选框的合并及NMS处理
def dect_box_handle(out_shape, test_img):
    box_list = []
    output_box = []
    length = len(out_shape)
    feature_map_handle(length, out_shape, test_img, box_list)
    # print box_list
    if box_list:
        retain_box_index = nms(np.array(box_list))
        for i in retain_box_index:
            output_box.append(box_list[i])
    return output_box

def forward_landmark(landmark_net, face_roi, bbox):
    ih, iw, _ = face_roi.shape
    sw = float(iw) / float(112)
    sh = float(ih) / float(112)

    blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1 / 127.5, size=(112, 112), mean = 127.5)
    landmark_net.setInput(blob)
    out = landmark_net.forward()
    points = out[0].flatten()
    for i in range(int(len(points) / 2)):
        points[i * 2] = ((points[i * 2] * 112) * sw) + bbox[0]
        points[(i * 2) + 1] = ((points[(i * 2) + 1] * 112) * sh) + bbox[1]
    return points

def draw_point(img, points):
    for i in range(int(len(points) / 2)):
        cv2.circle(img, (int(points[i * 2]), int(points[(i * 2) + 1])), 1, (255, 255, 255), 1)

if __name__ == "__main__":
    net = cv2.dnn.readNetFromCaffe('caffemodel/yoloface-50k.prototxt', 'caffemodel/yoloface-50k.caffemodel')
    landmark_net = cv2.dnn.readNetFromCaffe("caffemodel/landmark106.prototxt", "caffemodel/landmark106.caffemodel")

    imgpath = 'test.jpeg'
    srcimg = cv2.imread(imgpath)
    blob = cv2.dnn.blobFromImage(srcimg, scalefactor=1 / 256.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True)
    net.setInput(blob)
    outputs = net.forward()
    outputs = outputs.transpose(0, 3, 2, 1)[0,:]
    output_box = dect_box_handle(outputs, srcimg)

    for i in output_box:
        face_roi = srcimg[i[1]:i[3], i[0]:i[2]]
        points = forward_landmark(landmark_net, face_roi, i)
        draw_point(srcimg, points)
        print('detect', int(points.size * 0.5), 'points')
        cv2.rectangle(srcimg, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()