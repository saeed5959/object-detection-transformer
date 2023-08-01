import torch
import cv2
from einops import rearrange
import numpy as np
import json
from PIL import Image
from torchvision.models import EfficientNet_V2_M_Weights

from core.settings import model_config

def load_pretrained(model: object, pretrained: str, device: str):

    checkpoints = torch.load(pretrained, map_location=device)
    pretrained_model = checkpoints['model']
    step_all = checkpoints['step_all']
    epo = torch.tensor([checkpoints['epoch']]).to(device)
    lr = checkpoints['lr']
    model.load_state_dict(pretrained_model)

    return model, step_all, epo, lr

def img_preprocess_inference_old(img_path : str):
    img = cv2.imread(img_path) / 255
    img = cv2.resize(img,(256,256))

    img = rearrange(img, 'h w c -> c h w')
    img = torch.Tensor(img)
    img = img.unsqueeze(dim=0)

    return img

def img_preprocess_inference(img_path : str):
    weights= EfficientNet_V2_M_Weights.DEFAULT
    preprocess = weights.transforms()
    img = Image.open(img_path)
    img = preprocess(img).unsqueeze(dim=0)

    return img

def noraml_weight(file_path : str):
    
    with open(file_path) as file:
        data_file_in = file.readlines()


    category_dict = {}
    for data in data_file_in:
        data_patch = data.split("|")[1:]
        for patch in data_patch:
            id = int(patch.split(",")[0])
            if 0 < id <= 90:
                if id in category_dict:
                    category_dict[id] += 1
                else:
                    category_dict[id] = 1

    category_dict_sort = dict(sorted(category_dict.items(),key=lambda x:x[0]))

    weights = []
    category_dict_sum = sum(category_dict_sort.values())
    for counter in range(1,model_config.class_num+1):
        if counter in category_dict_sort:
            weights.append(category_dict_sum / category_dict_sort[counter])
        else:
            weights.append(0)

    weights = np.array(weights) / model_config.class_num
    weights_bound = np.minimum(10, np.maximum(0.1, weights))
    weights_bound = torch.Tensor(weights_bound)

    return weights_bound


def calculate_iou(box_a, box_b):
    #intersection over union
    cx_a, cy_a, w_a, h_a = box_a[0], box_a[1], box_a[2], box_a[3]
    cx_b, cy_b, w_b, h_b = box_b[0], box_b[1], box_b[2], box_b[3]
    a1_x, a1_y, a2_x, a2_y = (cx_a - w_a/2), (cy_a - h_a/2), (cx_a + w_a/2), (cy_a + h_a/2)
    b1_x, b1_y, b2_x, b2_y = (cx_b - w_b/2), (cy_b - h_b/2), (cx_b + w_b/2), (cy_b + h_b/2)

    overlap = max(0, min(a2_x,b2_x) - max(a1_x,b1_x)) * max(0, min(a2_y,b2_y) - max(a1_y,b1_y))
    mean_area = w_a*h_a + w_b*h_b - overlap
    if mean_area < 0.005:
        iou = 1
    else:
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
        class_id = np.argmax(class_out[patch])
        class_score = class_out[patch][class_id]
        if obj_score > model_config.obj_thresh and class_score > model_config.class_thresh and box_out[patch][2] > 0.03 and box_out[patch][3] > 0.03:
            obj_score_list.append(obj_score)

            x = patch % 2
            y = patch // 2
            xy_list.append((x,y))

            class_id = np.argmax(class_out[patch])
            class_list.append(class_id)
            
            class_score = class_out[patch][class_id]
            class_score_list.append(class_score)

            box_list.append(box_out[patch])

    # print(obj_score_list, class_list, class_score_list, box_list, xy_list)

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
        ref_id = class_list[max_score_index]
        shift = 0
        for count in range(len_box):
            possible_box = box_list[count- shift]
            iou = calculate_iou(ref_box, possible_box)
            if (iou > model_config.iou_thresh and ref_id==class_list[count-shift]) or count==max_score_index:
                del obj_score_list[count-shift]
                del class_list[count-shift]
                del class_score_list[count-shift]
                del box_list[count-shift]
                del xy_list[count-shift]
                shift += 1
                    

    return obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final


def show_box(img_path, class_list, box_list, out_path):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    color = (255,255,255)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    categories = category_name_id(model_config.json_file_path_1, model_config.json_file_path_2)
    for box, class_id in zip(box_list, class_list):
        class_name = categories[int(class_id)+1]
        cv2.rectangle(img, (int((box[0] - box[2]/2)*w), int((box[1] - box[3]/2)*h)), (int((box[0] + box[2]/2)*w), int((box[1] + box[3]/2)*h)), color, thickness)
        cv2.putText(img, class_name, (int(box[0]*w),int(box[1]*h)), font, fontScale, color, thickness, cv2.LINE_AA,)
    cv2.imwrite(out_path,img)

def category_name_id(file_path_1, file_path_2):
    with open(file_path_1) as file:
            data_file_in_1 = json.load(file)

    categories_1 = data_file_in_1["categories"]

    with open(file_path_2) as file:
                data_file_in_2 = json.load(file)

    categories_2 = data_file_in_2["categories"]

    category_dict = {}
    for data in categories_1:
        category_dict[data["id"]] = data["name"]

    for data in categories_2:
        category_dict[data["id"]] = data["name"]

    return category_dict    
