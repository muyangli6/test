import cv2
from ultralytics import YOLO
import os
import sys
from collections import defaultdict
import csv
from datetime import datetime
import time

def get_model_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

def list_cameras(max_tested=10):
    available_cams = []
    print("🔍 Detecting available cameras ...")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera {i} available")
            available_cams.append(i)
            cap.release()
        else:
            print(f"❌ Camera {i} not available")
    return available_cams

def save_counts_to_csv(class_counts, run_id, reset_type, filename="results.csv"):
    """Save counts to CSV with improved readability and a separating empty line"""
    if not class_counts:
        return
    file_exists = os.path.isfile(filename)
    total_count = sum(class_counts.values())
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["RunID", "Timestamp", "Class Name", "Count", "Percentage (%)", "Reset Type"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for cls, count in class_counts.items():
            percentage = round(count / total_count * 100, 2) if total_count > 0 else 0
            writer.writerow([run_id, timestamp, cls, count, percentage, reset_type])
        writer.writerow([])  # 空行分隔不同的计数块
    print(f"📁 Results saved to {filename} (RunID: {run_id}, ResetType: {reset_type})")

def rubbish_detect_track(model_path, video_path=0, conf_thresh=0.75, tracker_cfg="bytetrack.yaml", enable_record=True):
    """
    YOLOv8 + ByteTrack tracking + counting with recording functionality
    Keyboard controls:
      - Press '1': Save current counts to CSV & reset (manual)
      - Press 'q': Quit
    Auto-reset:
      - If NO OBJECTS are detected for NO_NEW_COUNT_THRESHOLD seconds
    """
    NO_NEW_COUNT_THRESHOLD = 5  # seconds

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Camera {video_path} failed to open")
        return

    # 获取视频帧尺寸和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20  # 默认 20 fps，如果获取不到

    # 初始化视频写入器
    if enable_record:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        print(f"🎥 Recording enabled. Saving to {video_filename}")
    else:
        out = None

    counted_ids = set()
    class_counts = defaultdict(int)
    last_seen_time = time.time()  # 最近一次检测到物体的时间
    run_id = 1  # Start Run ID

    instr_text = f"Press 1: Save & Reset Counts   q: Quit   Auto-reset: {NO_NEW_COUNT_THRESHOLD}s no objects"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        active_ids = set()
        count_updated = False

        # YOLO + Track
        results = list(model.track(frame, stream=True, persist=True,
                                   tracker=tracker_cfg, conf=conf_thresh))
        if results:
            r = results[0]
            try:
                annotated_frame = r.plot()
            except Exception:
                annotated_frame = frame.copy()

            if getattr(r, "boxes", None) is not None and r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                class_ids = r.boxes.cls.int().cpu().tolist()
                active_ids = set(track_ids)

                # 更新计数
                for track_id, cls_id in zip(track_ids, class_ids):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        cls_name = model.names[cls_id]
                        class_counts[cls_name] += 1
                        count_updated = True

        # 如果有检测到物体，就更新时间戳
        if active_ids:
            last_seen_time = time.time()

        # Draw counts + percentage on frame
        y_offset = 30
        total_count = sum(class_counts.values())
        for cls, count in class_counts.items():
            percentage = round(count / total_count * 100, 1) if total_count > 0 else 0
            cv2.putText(annotated_frame, f"{cls}: {count} ({percentage}%)", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            y_offset += 30

        # Draw instruction text
        h = annotated_frame.shape[0]
        cv2.putText(annotated_frame, instr_text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Draw current timestamp on frame
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, f"Time: {current_time_str}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Rubbish Detection + Tracking + Counting", annotated_frame)

        # 写入视频
        if enable_record and out is not None:
            out.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            save_counts_to_csv(class_counts, run_id, reset_type="manual")
            break
        elif key == ord('1'):
            save_counts_to_csv(class_counts, run_id, reset_type="manual")
            print(f"🔄 Counts manually reset (RunID: {run_id})")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            last_seen_time = time.time()

        # ✅ Auto-reset logic (只有“完全没有物体一段时间”才触发)
        if not active_ids and class_counts and (time.time() - last_seen_time) >= NO_NEW_COUNT_THRESHOLD:
            save_counts_to_csv(class_counts, run_id, reset_type="auto")
            print(f"⏱ Auto-reset: No objects detected for {NO_NEW_COUNT_THRESHOLD}s (RunID: {run_id})")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            last_seen_time = time.time()

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_file = get_model_path("best.pt")

    cams = list_cameras()
    if not cams:
        print("⚠️ No available cameras found")
        sys.exit()

    selected_cam = cams[0]
    print(f"📷 Using camera {selected_cam} for detection")

    rubbish_detect_track(model_path=model_file, video_path=selected_cam,
                         conf_thresh=0.65, tracker_cfg="bytetrack.yaml", enable_record=True)
