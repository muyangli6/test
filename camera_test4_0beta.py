import cv2
from ultralytics import YOLO
import os
import sys
import csv
from collections import OrderedDict
from datetime import datetime
import time
import numpy as np

# ✅ 使用无界面后端防止窗口缩小
import matplotlib
matplotlib.use('Agg')
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
NO_OBJECT_THRESHOLD = 10   # 🚀 没检测到任何物体多少秒后自动保存重置
WOOD_SIZE_RATIO = 0.10     # 占画面比例阈值：大于此视为长木头

COLOR_MAP = {
    "wood_large": (0, 0, 255),
    "wood_small": (0, 165, 255),
    "wood": (0, 255, 255),
    "default_box": (0, 255, 0),
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
        else:
            print(f"❌ 摄像头 {i} 无法打开")
    return available

def ensure_csv_header(filename):
    if not os.path.isfile(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["RunID", "Timestamp", "Class Name", "Count", "Percentage (%)", "Reset Type"])

# ================== CSV + 图表函数 ==================
def save_counts_to_csv_and_plot(class_counts, run_id, reset_type, filename=CSV_FILE):
    """保存CSV并生成白色背景饼图分析"""
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

    # ---- ✅ 白色背景饼图 ----
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    colors = plt.cm.tab20.colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')  # 白底
    ax.set_facecolor('white')
    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'alpha': 0.85}
    )
    ax.axis('equal')

    max_cls = max(class_counts, key=lambda k: class_counts[k])
    analysis_text = f"Most frequent: {max_cls}\n"
    for cls, count in class_counts.items():
        perc = round(count / total_count * 100, 1)
        analysis_text += f"{cls}: {count} ({perc}%)\n"
    fig.text(0.5, -0.05, analysis_text.strip(), ha='center', fontsize=8, color='black')

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"chart_{timestamp_str}.png"
    fig.savefig(plot_filename, bbox_inches='tight', transparent=False, facecolor='white')  # ✅ 强制白底
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
    run_id = 1
    last_seen_time = time.time()

    instr = f"Press 1: Save & Reset  q: Quit  (Auto-reset after {NO_OBJECT_THRESHOLD}s no objects)"
    print("▶ 检测开始，按 '1' 保存并重置，'q' 退出")

    window_name = "Rubbish Detection + Tracking + Counting"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 摄像头帧读取失败，重试中...")
            time.sleep(1)
            continue

        annotated = frame.copy()
        objects_present = False

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
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()

                if len(track_ids) > 0:
                    objects_present = True

                for tid, cls_id, box in zip(track_ids, class_ids, boxes_xyxy):
                    cls_name = model.names[cls_id]

                    # 🪵 按面积判断长木头 / 短木头
                    if cls_name == "wood":
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        frame_area = width * height
                        cls_name = "wood_large" if area >= frame_area * WOOD_SIZE_RATIO else "wood_small"

                    if tid not in counted_ids:
                        counted_ids.add(tid)
                        class_counts.setdefault(cls_name, 0)
                        class_counts[cls_name] += 1

        # ✅ 更新最后看到物体的时间
        if objects_present:
            last_seen_time = time.time()

        # ✅ 若超过指定时间未检测到物体 → 自动保存+清零
        if class_counts and not objects_present and (time.time() - last_seen_time) >= NO_OBJECT_THRESHOLD:
            save_counts_to_csv_and_plot(class_counts, run_id, "auto")
            print(f"⏱ 超过 {NO_OBJECT_THRESHOLD}s 未检测到任何物体 -> 自动重置 (RunID={run_id})")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            last_seen_time = time.time()
            cv2.resizeWindow(window_name, 1280, 720)  # 💪 保持窗口大小

        # 左上角统计
        total = sum(class_counts.values())
        lines = [f"{cls}: {cnt} ({round(cnt/total*100,1) if total else 0}%)"
                 for cls, cnt in class_counts.items()]
        draw_text_multiline(annotated, lines, 10, 30, 26, (0,255,0), 0.8, 2)

        # 底部说明
        h = annotated.shape[0]
        draw_text_multiline(annotated, [instr], 10, h - 30, 22, COLOR_MAP["text_instr"], 0.6)
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
            print(f"🔄 手动重置计数 (RunID={run_id})")
            run_id += 1
            class_counts.clear()
            counted_ids.clear()
            last_seen_time = time.time()
            cv2.resizeWindow(window_name, 1280, 720)  # 💪 防止窗口缩小

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
