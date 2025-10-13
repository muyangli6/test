import cv2
from ultralytics import YOLO
import os
import sys
from collections import defaultdict, OrderedDict
import csv
from datetime import datetime
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止窗口缩小
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# ================== CONFIG ==================
MODEL_PATH = "best.pt"
CSV_FILE = "results.csv"
OUTPUT_VIDEO_DIR = "output_videos"

CONF_THRESH = 0.65
TRACKER_CFG = "bytetrack.yaml"
NO_OBJECT_THRESHOLD = 10   # 未检测到物体多少秒自动保存+重置
MIN_FRAMES_FOR_COUNT = 3   # 连续出现 N 帧才计数

COLOR_MAP = {
    "text_time": (0, 255, 255),
    "text_instr": (255, 255, 255)
}

# ================== HELPERS ==================
def get_model_path(filename):
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

def list_cameras(max_tested=10):
    available = []
    print("🔍 正在检测可用摄像头...")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"✅ 摄像头可用: {i}")
            cap.release()
    return available

def ensure_csv_header(filename):
    if not os.path.isfile(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["RunID", "Timestamp", "Class Name", "Count", "Percentage (%)", "Reset Type"])

def save_counts_to_csv_and_plot(class_counts, run_id, reset_type, filename=CSV_FILE):
    """保存CSV并生成饼图"""
    if not class_counts:
        return None

    ensure_csv_header(filename)
    total_count = sum(class_counts.values())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for cls, count in class_counts.items():
            perc = round(count / total_count * 100, 2) if total_count > 0 else 0.0
            writer.writerow([run_id, timestamp, cls, count, perc, reset_type])
        writer.writerow([])

    print(f"📁 已保存CSV: {filename} (RunID={run_id}, Reset={reset_type})")

    # ---- 生成饼图 ----
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    colors = plt.cm.tab20.colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')

    max_cls = max(class_counts, key=lambda k: class_counts[k])
    analysis_text = f"Most frequent: {max_cls}\n"
    for cls, count in class_counts.items():
        perc = round(count / total_count * 100, 1)
        analysis_text += f"{cls}: {count} ({perc}%)\n"
    fig.text(0.5, -0.05, analysis_text.strip(), ha='center', fontsize=8)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"chart_{timestamp_str}.png"
    fig.savefig(plot_filename, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"📊 饼图已保存: {plot_filename}")
    return plot_filename

def draw_text_multiline(img, lines, x, y, line_height=22, color=(255,255,255), scale=0.6, thickness=1):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ================== MAIN ==================
def rubbish_detect_track(model_path, video_source):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {video_source}")
        return

    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_out = os.path.join(OUTPUT_VIDEO_DIR, f"output_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))
    print(f"🎥 开始录制: {video_out}")

    counted_ids = set()
    class_counts = OrderedDict()
    track_frames = defaultdict(int)
    run_id = 1
    last_seen_time = time.time()
    trajectories = defaultdict(list)  # 存放每个 track_id 的轨迹点 [(x, y), ...]

    instr = f"Press 1: Save & Reset  q: Quit  (Auto-reset after {NO_OBJECT_THRESHOLD}s no objects)"

    window_name = "Rubbish Detection + Tracking + Counting"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        active_ids = set()

        results = list(model.track(frame, stream=True, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESH))
        if results:
            r = results[0]
            try:
                annotated = r.plot()
            except:
                annotated = frame.copy()

            if getattr(r, "boxes", None) is not None and r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                class_ids = r.boxes.cls.int().cpu().tolist()
                bboxes = r.boxes.xyxy.cpu().numpy().tolist()
                active_ids = set(track_ids)

                for tid, cls_id, bbox in zip(track_ids, class_ids, bboxes):
                    track_frames[tid] += 1
                    # 更新轨迹
                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)
                    trajectories[tid].append((x_center, y_center))
                    if len(trajectories[tid]) > 30:  # 最多保存最近 30 个点
                        trajectories[tid] = trajectories[tid][-30:]

                    # 绘制轨迹
                    for i in range(1, len(trajectories[tid])):
                        cv2.line(annotated, trajectories[tid][i-1], trajectories[tid][i], (0,255,255), 2)

                    if track_frames[tid] >= MIN_FRAMES_FOR_COUNT and tid not in counted_ids:
                        counted_ids.add(tid)
                        cls_name = model.names[cls_id]
                        class_counts.setdefault(cls_name, 0)
                        class_counts[cls_name] += 1

        if active_ids:
            last_seen_time = time.time()

        # 左上角显示计数
        total = sum(class_counts.values())
        y_offset = 30
        for cls, cnt in class_counts.items():
            perc = round(cnt / total * 100, 1) if total else 0
            cv2.putText(annotated, f"{cls}: {cnt} ({perc}%)", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            y_offset += 30

        # 底部说明
        h = annotated.shape[0]
        cv2.putText(annotated, instr, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP["text_instr"], 2)
        cv2.putText(annotated, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP["text_time"], 2)

        out_writer.write(annotated)
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if class_counts:
                save_counts_to_csv_and_plot(class_counts, run_id, "manual")
            break
        elif key == ord('1'):
            if class_counts:
                save_counts_to_csv_and_plot(class_counts, run_id, "manual")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            track_frames.clear()
            trajectories.clear()
            last_seen_time = time.time()
            cv2.resizeWindow(window_name, 1280, 720)

        # 自动保存 & 重置
        if not active_ids and class_counts and (time.time() - last_seen_time) >= NO_OBJECT_THRESHOLD:
            save_counts_to_csv_and_plot(class_counts, run_id, "auto")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            track_frames.clear()
            trajectories.clear()
            last_seen_time = time.time()
            cv2.resizeWindow(window_name, 1280, 720)

        # 清理消失的 track_id
        disappeared = [tid for tid in track_frames if tid not in active_ids]
        for tid in disappeared:
            track_frames.pop(tid)
            trajectories.pop(tid, None)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"🎬 视频已保存: {video_out}")
    print("✅ 程序结束。")

# ================== ENTRY ==================
if __name__ == "__main__":
    cams = list_cameras()
    if not cams:
        print("⚠️ 未检测到可用摄像头")
        sys.exit(1)
    cam = cams[0]
    print(f"📷 使用摄像头: {cam}")
    model_file = get_model_path(MODEL_PATH)
    rubbish_detect_track(model_file, cam)
