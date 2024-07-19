from ultralytics import YOLO

model = YOLO('/home/data3/zys/results/DroneVehicle/yolov8l-obb_ir_e50_bs8/weights/best.pt')
# count = 0
# count_lora = 0
# for n, p in model.model.named_parameters():
#     count += p.numel()
#     # print(n)
#     # print(p.numel())
#     if 'lora' in n:
#         count_lora += p.numel()
# print(count)
# print(count_lora)

# count = 0
# for n, p in model.model.named_parameters():
#     # print(n)
#     if 'lora_A' in n:
#         s_a = p.shape[1]
#     if 'lora_E' in n:
#         s_e = (p != 0).sum()
#         # s_e = p.shape[0]
#     if 'lora_B' in n:
#         s_b = p.shape[0]
#         count += ((s_a * s_e) + (s_b * s_e) + s_e)
# print(count)

data = 'drone_vehicle.yaml'
model.val(data=data,
          project='DroneVehicle',
          name='val_yolov8l-obb_ir_e50_bs8',
          imgsz=800,
          batch=8,
          device=7,
          # split='val_rgb',
          )