import torch
import cv2
from einops import rearrange
import numpy as np

from core.settings import model_config

def load_pretrained(model: object, pretrained: str, device: str):

    pretrained_model = torch.load(pretrained, map_location=device)
    model.load_state_dict(pretrained_model)

    return model

def img_preprocess_inference(img_path : str):
    img = cv2.imread(img_path) / 255
    img = cv2.resize(img,(256,256))

    img = rearrange(img, 'h w c -> c h w')
    img = torch.Tensor(img)
    img = img.unsqueeze(dim=0)

    return img


def calculate_iou(box_a, box_b):
    #intersection over union
    cx_a, cy_a, w_a, h_a = box_a[0], box_a[1], box_a[2], box_a[3]
    cx_b, cy_b, w_b, h_b = box_b[0], box_b[1], box_b[2], box_b[3]
    a1_x, a1_y, a2_x, a2_y = (cx_a - w_a/2), (cy_a - h_a/2), (cx_a + w_a/2), (cy_a + h_a/2)
    b1_x, b1_y, b2_x, b2_y = (cx_b - w_b/2), (cy_b - h_b/2), (cx_b + w_b/2), (cy_b + h_b/2)

    overlap = max(0, min(a2_x,b2_x) - max(a1_x,b1_x)) * max(0, min(a2_y,b2_y) - max(a1_y,b1_y))
    mean_area = w_a*h_a + w_b*h_b - overlap

    iou = overlap / mean_area

    return iou


def nms_img(obj_out, class_out, box_out):
    #non maxima supression 
    obj_score_list = []
    class_list = []
    class_score_list = []
    box_list = []
    xy_list = []

    #making possible bbox
    for patch in range(len(obj_out)):
        obj_score = obj_out[patch]
        if  obj_score > 0.5:
            obj_score_list.append(obj_score)

            x = patch % 2
            y = patch // 2
            xy_list.append((x,y))

            class_id = np.argmax(class_out[patch])
            class_list.append(class_id)
            
            class_score = class_out[patch][class_id]
            class_score_list.append(class_score)

            box_list.append(box_out[patch])

    print(obj_score_list, class_list, class_score_list, box_list, xy_list)

    obj_score_list_final = []
    class_list_final = []
    class_score_list_final = []
    box_list_final = []
    xy_list_final = []

    while obj_score_list != []:
        max_score_index = np.argmax(obj_score_list)
        obj_score_list_final.append(obj_score_list[max_score_index])
        class_list_final.append(class_list[max_score_index])
        class_score_list_final.append(class_score_list[max_score_index])
        box_list_final.append(box_list[max_score_index])
        xy_list_final.append([xy_list[max_score_index]])

        len_box = len(box_list)
        ref_box = box_list[max_score_index]
        shift = 0
        for count in range(len_box):
            possible_box = box_list[count- shift]

            iou = calculate_iou(ref_box, possible_box)
            if iou > model_config.iou_thresh:
                del obj_score_list[count-shift]
                del class_list[count-shift]
                del class_score_list[count-shift]
                del box_list[count-shift]
                del xy_list[count-shift]
                shift += 1
                    

    return obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final

# obj_out = np.array([1,0,0.6,0.7])
# class_out = np.array([[0,1,0,0.5,0.3],[0,0,1,0,0],[0,0,0,1,0.3],[0,0,0,1,0.4]])
# box_out = np.array([[0.2,0.2,0.1,0.1],[0,0,0,0.1],[0.5,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]])
# print(nms_img(obj_out, class_out, box_out))

def show_box(img_path, class_list, box_list):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    color = (255,255,255)
    thickness = 2
    for box in box_list:
        cv2.rectangle(img, (int((box[0] - box[2]/2)*w), int((box[1] - box[3]/2)*h)), (int((box[0] + box[2]/2)*w), int((box[1] + box[3]/2)*h)), color, thickness)

    cv2.imwrite("./out.jpg",img)