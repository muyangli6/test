import os

def seg_to_det(seg_txt, det_txt, img_width, img_height):
    with open(seg_txt, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        cls = parts[0]

        # YOLO-seg 格式: class cx cy w h x1 y1 x2 y2 ...
        # 前 5 个是 cls + bbox，后面是分割点
        coords = list(map(float, parts[5:]))

        if len(coords) < 6:
            continue  # 没有分割点，跳过

        xs = coords[0::2]
        ys = coords[1::2]

        # 找到分割点的边界框
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # 转成 YOLO-detect 格式 (cx, cy, w, h)，并归一化
        cx = ((xmin + xmax) / 2) / img_width
        cy = ((ymin + ymax) / 2) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        new_line = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        new_lines.append(new_line)

    # 写入新文件
    with open(det_txt, "w") as f:
        f.writelines(new_lines)


def batch_convert(seg_dir, det_dir, img_dir):
    """
    seg_dir: 分割标注目录 (labels)
    det_dir: 转换后检测标注目录 (labels_detect)
    img_dir: 图片目录 (images)，用于获取图像宽高
    """
    import cv2
    os.makedirs(det_dir, exist_ok=True)

    for file in os.listdir(seg_dir):
        if not file.endswith(".txt"):
            continue

        seg_path = os.path.join(seg_dir, file)
        img_path_jpg = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        img_path_png = os.path.join(img_dir, file.replace(".txt", ".png"))

        # 找对应的图片获取尺寸
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            print(f"⚠️ 找不到图片 {file}, 跳过")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 打不开图片 {img_path}, 跳过")
            continue
        h, w = img.shape[:2]

        det_path = os.path.join(det_dir, file)
        seg_to_det(seg_path, det_path, w, h)
        print(f"✅ 转换完成: {file}")


if __name__ == "__main__":
    # 你需要改这三个路径
    seg_labels_dir = "C:/Users/Yorushika/Desktop/dataset_and_picture/ALL/together/labels/"      # 原分割标签路径
    det_labels_dir = "C:/Users/Yorushika/Desktop/dataset_and_picture/ALL/together/output/"   # 输出检测标签路径
    images_dir = "C:/Users/Yorushika/Desktop/dataset_and_picture/ALL/together/images/"              # 图片路径

    batch_convert(seg_labels_dir, det_labels_dir, images_dir)
    print("🎉 所有标注已转换完成！")
