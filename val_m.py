from ultralytics import YOLO_m

model = YOLO_m('/home/zhangyusi/DroneVehicle/yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8/weights/best.pt')
# model = YOLO_m('/home/zhangyusi/DroneVehicle/yolov8l-obb_e50_bs8/weights/best.pt')
# count = 0
# count_lora = 0
# for n, p in model.model.named_parameters():
#     count += p.numel()
#     print(n)
#     # print(p.numel())
#     if 'lora' in n:
#         count_lora += p.numel()
# print(count)
# print(count_lora)

# count = 0
# for n, p in model.model.named_parameters():
#     # print(n)
#     if 'lora_A_rgb' in n:
#         s_a_rgb = p.shape[1]
#     if 'lora_A_ir' in n:
#         s_a_ir = p.shape[1]
#     if 'lora_E' in n:
#         s_e = (p != 0).sum()
#         print(n)
#         print(s_e)
#     # if 'lora_E_rgb' in n:
#     #     s_e_rgb = (p != 0).sum()
#     #     print(n)
#     #     print(s_e_rgb)
#     #     # s_e = p.shape[0]
#     # if 'lora_E_ir' in n:
#     #     s_e_ir = (p != 0).sum()
#     #     print(n)
#     #     print(s_e_ir)
#     if 'lora_B_rgb' in n:
#         s_b_rgb = p.shape[0]
#     if 'lora_B_ir' in n:
#         s_b_ir = p.shape[0]
#         count += ((s_a_rgb * s_e) + (s_b_rgb * s_e) + (s_a_ir * s_e) + (s_b_ir * s_e) + s_e)
#         # count += ((s_a_rgb * s_e_rgb) + (s_b_rgb * s_e_rgb) + \
#         #           (s_a_ir * s_e_ir) + (s_b_ir * s_e_ir) + s_e_rgb + s_e_ir)
# print(count)


# rank_count = [[], [], [], [], [], [], [], [], [], []]
# for n, p in model.model.named_parameters():
#     module_index = int(n.split('.')[1])
#     if 'lora_E' in n:
#         if module_index < 10:
#             s_e = (p != 0).sum()
#             rank_count[module_index].append(s_e)
#         else:
#             raise ValueError('what?')
# for i in range(10):
#     print(f'num {i} module rank')
#     print(sum(rank_count[i])/len(rank_count[i]))

data = 'drone_vehicle_m.yaml'
model.val(data=data,
          project='DroneVehicle',
          name='val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best',
          imgsz=800,
          batch=1,
          device=0,
          visualize=True,
          # split='val_rgb',
          )