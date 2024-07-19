from ultralytics import YOLO

model = YOLO("yolov8l.yaml")
model.train(data="LLVIP.yaml",
            epochs=50,
            patience=30,
            batch=8,
            imgsz=640,
            device=2,
            # r_init=24,
            # r_target=6,
            # adalora=False,
            project="LLVIP",
            name='yolov8l_ir_e50_bs8',
            resume=False,
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            )