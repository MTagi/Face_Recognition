Dự án "Face Recognition" là một ứng dụng nhận diện khuôn mặt được phát triển bằng Python, sử dụng các thư viện như OpenCV và Dlib. Mục tiêu của dự án là cung cấp một công cụ mạnh mẽ để phát hiện và nhận diện khuôn mặt trong hình ảnh và video từ đó điểm danh trong cơ sở dữ liệu.

Công nghê: Sử dụng YOLOv8, iResNet và cơ sở dữ liệu MongoDB. 

Các bước nhận diện: 
1. Detect các khuôn mặt dựa trên hình ảnh.
2. Embedding các khuôn mặt và so sánh với các embedding trong database.
3. Nhận diện các khuông mặt nếu có sự trùng khớp giữa 2 embedding. 

Run: 
1. Cài đặt các thư viện cần thiết.  
2. Go live file index.html để thực hiện chạy giao diện.
3. Chạy file backend.py để thực hiện chạy backend kết nối với fontend.
4. Thêm các user trên giao diện.
5. Chạy file main.py để bật camera và bắt đầu điểm danh.  
