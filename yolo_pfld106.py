import cv2
from detect_img import dect_box_handle, draw_point

INPUT_SIZE = 56
def forward_landmark(landmark_net, face_roi, bbox):
    ih, iw, _ = face_roi.shape
    sw = float(iw) / float(112)
    sh = float(ih) / float(112)

    blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1 / 255.0, size=(112, 112))
    landmark_net.setInput(blob)
    out = landmark_net.forward()
    points = out[0].flatten()
    for i in range(int(len(points) / 2)):
        points[i * 2] = ((points[i * 2] * 112) * sw) + bbox[0]
        points[(i * 2) + 1] = ((points[(i * 2) + 1] * 112) * sh) + bbox[1]
    return points

if __name__ == "__main__":
    net = cv2.dnn.readNetFromCaffe('caffemodel/yoloface-50k.prototxt', 'caffemodel/yoloface-50k.caffemodel')
    landmark_net = cv2.dnn.readNet('v3.onnx')

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
        print('detect', int(points.size*0.5), 'points')
        cv2.rectangle(srcimg, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()