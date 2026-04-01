import cv2

def list_cameras(max_tested=10):
    """
    测试并列出可用的摄像头设备编号
    """
    available_cams = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows 建议加 CAP_DSHOW
        if cap.isOpened():
            print(f"✅ Camera {i} is available")
            available_cams.append(i)
            cap.release()
        else:
            print(f"❌ Camera {i} not available")
    return available_cams


if __name__ == "__main__":
    cams = list_cameras()
    if cams:
        print("\n可用摄像头编号:", cams)
        # 打开第一个可用摄像头测试一下
        cap = cv2.VideoCapture(cams[0])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("⚠️ 没有找到可用的摄像头")
