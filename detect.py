import torch
from ultralytics import YOLO
from torchvision import transforms
import cv2
# from embedding_face import get_feature  # Giả sử bạn có sẵn hàm này
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

size_convert = 640  # Kích thước chuẩn để đưa qua model
conf_thres = 0.4
iou_thres = 0.5

face_preprocess = transforms.Compose([
    transforms.ToTensor(),  # Input PIL => (3, 56, 56), /255.0
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = YOLO("E:/3.hocki1nam4/xu li anh/face_recognition/weights/yolov8n-face.pt")
# def preprocess_image(image, img_size=640):

    # h, w = image.shape[:2]
    # r = img_size / max(h, w)  # Tỷ lệ scale
    # new_h, new_w = int(h * r), int(w * r)
    # image_resized = cv2.resize(image, (new_w, new_h))
# 

    # top = (img_size - new_h) // 2
    # bottom = img_size - new_h - top
    # left = (img_size - new_w) // 2
    # right = img_size - new_w - left
    # image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


    # image_padded = image_padded[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB và HWC -> CHW
    # image_padded = image_padded.copy()  # Tạo bản sao của mảng để tránh stride tiêu cực
    # image_tensor = torch.from_numpy(image_padded).float() / 255.0  # Chuẩn hóa về [0, 1]
    # return image_tensor.unsqueeze(0)  # Thêm chiều Batch
@torch.no_grad()
def detect_face(image_face):
    # img = preprocess_image(image_face, size_convert)
    with torch.no_grad():
        result = model.predict(image_face, conf=conf_thres, iou=iou_thres)
    return result

# def get_face_embeddings(image_face, pre_trained_embedding=None):
    # result = detect_face(image_face)
    # 
    # face_embeddings = []
    # for box in result[0].boxes.xyxy:  # Duyệt qua từng khuôn mặt phát hiện
        # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # face_crop = image_face[y1:y2, x1:x2]
        # 
        # face_embedding = get_feature(face_crop)  # Hàm này giả định đã lấy embedding khuôn mặt
        # 
        # Thêm feature vào danh sách
        # face_embeddings.append(face_embedding)
        # 
        # Vẽ bounding box và in kết quả lên ảnh
        # cv2.rectangle(image_face, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box màu xanh
        # sims = np.dot(face_embedding, pre_trained_embedding.T)
        # label = f"Similarity: {sims[0][0]:.2f}" if pre_trained_embedding is not None else "Face"
        # cv2.putText(image_face, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # In kết quả lên trên bbox
    # 
    # return face_embeddings

# image=cv2.imread("E:/3.hocki1nam4/xu li anh/face_recognition/data/thang.png")
   # Giả sử bạn đã có sẵn một vector embedding cho một người nào đó
# pre_trained_embedding = get_feature(image)  # Đây chỉ là một ví dụ
# Mở camera (thường sẽ là thiết bị 0 nếu chỉ có một camera)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
    # print("Không thể mở camera")
    # exit()

# while True:
    # Đọc từng frame từ camera
    # ret, frame = cap.read()
    # if not ret:
        # print("Không đọc được frame từ camera")
        # break

    #   Lấy embedding và vẽ bbox lên ảnh
    # get_face_embeddings(frame, pre_trained_embedding)

    #   Hiển thị ảnh với bounding boxes
    # cv2.imshow("Detected Faces", frame)

    # Nếu nhấn phím 'q' thì thoát
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()

# print("Face Embeddings: ", face_embeddings)
# print("Cosine Similarities: ", similarities)
