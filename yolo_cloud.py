model = YOLO('/home/yorushika/ultralytics/ultralytics/models/v8/yolov8x.yaml')
model.train(
    data='/home/yorushika/ultralytics/yolo-bvn.yaml',
    workers=0,
    epochs=300,
    batch=16,
    imgsz=640,
    patience=50,
    device=0
)

