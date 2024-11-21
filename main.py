import cv2
import numpy as np
from embedding_face import get_feature
from detect import detect_face
from recognize import attendance_check


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()
while True:
    # Đọc từng frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ camera")
        break
    result = detect_face(frame)
    for box in result[0].boxes.xyxy:  # Duyệt qua từng khuôn mặt phát hiện
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face_crop = frame[y1:y2, x1:x2]
        face_embedding = get_feature(face_crop)  # Hàm này giả định đã lấy embedding khuôn mặt
        result = attendance_check(face_embedding)
        if result != "Không xác định":
            label = f"{result['name']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box màu xanh
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # In kết quả lên trên bbox
        else: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vẽ bounding box màu đỏ
    cv2.resize(frame, (640, 640))
    cv2.imshow("Detected Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

