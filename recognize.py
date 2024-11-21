import pymongo
import numpy as np

# Hàm kết nối MongoDB
def get_mongo_db():
    client = pymongo.MongoClient("mongodb://localhost:27017/")  # Thay đổi URL nếu cần
    db = client["xulianh"]  # Thay 'your_database_name' bằng tên cơ sở dữ liệu của bạn
    return db["mtagi"]  # Thay 'your_collection_name' bằng tên collection của bạn

# Hàm so sánh vector
def compare_encodings(encoding, encodings):
    sims = np.dot(encodings, encoding.T)  # Tính độ tương đồng (cosine similarity)
    pare_index = np.argmax(sims)  # Lấy chỉ số của vector có độ tương đồng cao nhất
    score = sims[pare_index]  # Lấy độ tương đồng
    return score, pare_index

# Hàm điểm danh và so sánh vector với cơ sở dữ liệu MongoDB
def attendance_check(encoding):
    # Lấy dữ liệu vector từ MongoDB
    collection = get_mongo_db()
    records = collection.find({})  # Lấy tất cả các bản ghi trong collection

    # Chuyển đổi các vector trong database thành một list numpy
    encodings = []
    student_ids = []
    for record in records:
        encodings.append(np.array(record["embedding_face"]))  # Vector đặc trưng được lưu dưới key "embedding_face"
        student_ids.append(record["mssv"])  # Mã sinh viên lưu dưới key "mssv"

    encodings = np.array(encodings)

    # So sánh vector mới với các vector trong cơ sở dữ liệu
    score, pare_index = compare_encodings(encoding, encodings)

    if score > 0.3:  # Nếu độ chính xác lớn hơn 0.3
        # Cập nhật trạng thái từ "chưa điểm danh" sang "đã điểm danh"
        student_id = student_ids[pare_index]
        collection.update_one(
            {"mssv": student_id},
            {"$set": {"status": "đã điểm danh"}}  # Cập nhật trạng thái
        )
        
        # Lấy thông tin sinh viên
        student_info = collection.find_one({"mssv": student_id})
        return {
            "mssv": student_info["mssv"],
            "name": student_info["name"]  # Tên sinh viên lưu dưới key "name"
        }
    else:
        return "Không xác định"