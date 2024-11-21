from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import os
from PIL import Image
import numpy as np
import uuid
from flask_cors import CORS
from detect import detect_face
from embedding_face import get_feature
import cv2
app = Flask(__name__)
CORS(app)

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["xulianh"]
collection = db["mtagi"]

# Hàm tạo embedding_face từ ảnh
def create_embedding(image_path):
    image_face = cv2.imread(image_path)
    result = detect_face(image_face)
    x1, y1, x2, y2 = int(result[0].boxes.xyxy[0][0]), int(result[0].boxes.xyxy[0][1]), int(result[0].boxes.xyxy[0][2]), int(result[0].boxes.xyxy[0][3])
    face_crop = image_face[y1:y2, x1:x2]
    face_embedding = get_feature(face_crop)
    return face_embedding


# API thêm sinh viên
@app.route('/add_student', methods=['POST'])
def add_student():
    # Nhận dữ liệu từ frontend
    mssv = request.form['mssv']
    name = request.form['name']
    image = request.files['image']
    
    # Kiểm tra đầy đủ thông tin
    if not mssv or not name or not image:
        return jsonify({"status": "error", "message": "Vui lòng nhập đầy đủ thông tin!"})

    # Lưu ảnh vào thư mục data với tên MSSV
    image_path = f"data/{mssv}.jpg"
    if not os.path.exists("data"):
        os.makedirs("data")
    image.save(image_path)

    # Tạo embedding face
    embedding_face = create_embedding(image_path)
    if embedding_face is None:
        return jsonify({"status": "error", "message": "Không thể tạo embedding face!"})

    # Lưu thông tin vào MongoDB
    student_data = {
        "mssv": mssv,
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "chưa điểm danh",
        "embedding_face": embedding_face.tolist()  # Chuyển numpy array thành list để lưu vào MongoDB
    }
    collection.insert_one(student_data)

    return jsonify({"status": "success", "message": "Thêm sinh viên thành công!"})

# API lấy danh sách sinh viên
@app.route('/students', methods=['GET'])
def get_students():
    # Lấy dữ liệu sinh viên từ MongoDB
    date = request.args.get('date', default=datetime.now().strftime("%Y-%m-%d"))
    students = collection.find({"date": date})
    
    student_list = []
    for student in students:
        student_list.append({
            "mssv": student['mssv'],
            "name": student['name'],
            "date": student['date'],
            "status": student['status']
        })
    print(student_list)
    return jsonify(student_list)

# Trang chủ hiển thị form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
