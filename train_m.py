from ultralytics import YOLO_m

model = YOLO_m("yolov8n-obb_lma.yaml")
model.train(data="drone_vehicle_m.yaml",
            epochs=50,
            patience=30,
            batch=8,
            imgsz=800,
            device=3,
            r_init=9,
            r_target=6,
            adalora=True,
            project="DroneVehicle",
            name='yolov8n_obb_lma_r9-6_e50_bs8',
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            )
