import json
import glob
from pathlib import Path

# 输入 LabelMe JSON 文件路径
input_path = r"C:/Users/Yorushika/Desktop/jpg+json/*.json"
# 输出 YOLOv8-seg 标签路径
output_path = r"C:/Users/Yorushika/Desktop/wawawa/"

# 类别字典（根据你的数据集修改）
category_to_id = {'wood': 0, 'pipe': 1, 'brick': 2, 'cardboard': 3,'plastic bag':4,'bottle':5}

files = glob.glob(input_path)

for file in files:
    name = Path(file).stem
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in category_to_id:
            continue  # 跳过未定义类别

        class_id = category_to_id[label]
        points = shape["points"]  # 多边形点集

        # 归一化并展开成 [x1, y1, x2, y2, ...]
        normalized_points = []
        for x, y in points:
            nx = round(x / w, 6)
            ny = round(y / h, 6)
            normalized_points.extend([nx, ny])

        # 拼接 YOLOv8-seg 一行
        line = [str(class_id)] + [str(p) for p in normalized_points]
        yolo_lines.append(" ".join(line))

    # 保存到 txt
    with open(Path(output_path) / f"{name}.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(yolo_lines))

print("✅ 转换完成，已生成 YOLOv8-seg 兼容的标签文件！")
