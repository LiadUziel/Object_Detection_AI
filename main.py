import cv2

thresh = 0.60  # Threshold to detect object (Accuracy percentages)

capture_video = cv2.VideoCapture(1)  # 1 for DroidCam
capture_video.set(3, 640)
capture_video.set(4, 480)

# Initial detection settings
class_names = []
class_file = "objects.names"
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize((320, 320))
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# End initial detection settings

while True:  # if image - remove this
    sucess, img = capture_video.read()  # if image - remove this

    class_Ids, confs, bbox = net.detect(img, confThreshold=thresh)
    print(class_Ids, bbox)

    if len(class_Ids) != 0:
        for class_ID, confidence, box in zip(class_Ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Object name
            cv2.putText(img, class_names[class_ID - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

            # Accuracy percentages
            cv2.putText(img, str(round(confidence * 100, 2)) + '%', (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", img)

    cv2.waitKey(1)  # cv2.waitKey(0)

    if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

    # End of loop
