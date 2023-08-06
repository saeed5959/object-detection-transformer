import argparse
import numpy as np
import cv2
import torch
import os
import json
from tqdm import tqdm

from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img, show_box, calculate_iou
from core.settings import train_config, model_config

device = train_config.device


def compare(class_gt, box_gt, class_out, box_out, img_path, img_out_gt_path, intersection_class_out_count, class_out_count):

    img = cv2.imread(img_path)
    h, w, c = img.shape

    gt_count = len(class_gt)
    out_count = len(class_out)
    intersection_count = 0
    all_class_count = 0
    box_gt_norm = []

    for obj_id_in, obj_box_in in zip(class_gt, box_gt):
        # if int(obj_id_in)==1 :
        #     all_class_count += 1
        obj_box_in[0], obj_box_in[1], obj_box_in[2], obj_box_in[3] = obj_box_in[0]/w, obj_box_in[1]/h, obj_box_in[2]/w, obj_box_in[3]/h
        obj_box_in[0] = obj_box_in[0] + obj_box_in[2]/2
        obj_box_in[1] = obj_box_in[1] + obj_box_in[3]/2
        
        box_gt_norm.append(obj_box_in)

        for obj_id_out, obj_box_out in zip(class_out, box_out):
            if int(obj_id_in)==int(obj_id_out+1) and calculate_iou(obj_box_in, obj_box_out) > model_config.iou_thresh_dataset:
                intersection_count += 1
                intersection_class_out_count[int(obj_id_out)] += 1
                break

    for obj_id_out, obj_box_out in zip(class_out, box_out):
        class_out_count[int(obj_id_out)] += 1


    class_gt_norm = [id-1 for id in class_gt]
    #show_box(img_path, class_gt_norm, box_gt_norm, img_out_gt_path)

    return gt_count, intersection_count, out_count


def inference_img(img_path, model, img_out_path):

    #prepare input image
    img = img_preprocess_inference(img_path)
    img = img.to(device)
    
    poa = []
    epoch = torch.tensor([30]).to(device)
    #giving input to model
    obj_out, class_out, box_out = model.inference(img, poa, epoch)
    
    obj_out, class_out, box_out = obj_out[0].detach().cpu().numpy(), class_out[0].detach().cpu().numpy(), box_out[0].detach().cpu().numpy()

    obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final = nms_img(obj_out, class_out, box_out)

    # show_box(img_path, class_list_final, box_list_final, img_out_path)

    return class_list_final, box_list_final


def inference(folder_in_path: str, model_path: str, folder_out_path: str, folder_out_gt_path: str, ground_truth_file: str):

    #load model
    model = models.VitModel().to(device)
    model, step_all, epo, lr = load_pretrained(model, model_path, device)
    model.eval()

    with open(ground_truth_file) as file:
        ground_truth = json.load(file)["annotations"]

    all_gt_count = 0
    all_out_count = 0
    intersection_count = 0
    intersection_class_out_count = np.zeros(90)
    class_out_count = np.zeros(90)

    for data in tqdm(ground_truth):
        
        img_path = os.path.join(folder_in_path, data["file_name"][:-3] + "jpg")
        img_out_path = os.path.join(folder_out_path, data["file_name"][:-3] + "jpg")
        img_out_gt_path = os.path.join(folder_out_gt_path, data["file_name"][:-3] + "jpg")

        print(img_path)

        class_gt = []
        box_gt = []
        for obj in data["segments_info"]:
            if obj["category_id"] < 91:
                class_gt.append(obj["category_id"])
                box_gt.append(obj["bbox"])

        class_out, box_out = inference_img(img_path, model, img_out_path)

        gt_count_img, intersection_count_img, out_count_img = compare(class_gt, box_gt, class_out, box_out, img_path, img_out_gt_path,
                                                                       intersection_class_out_count, class_out_count)

        print(gt_count_img, intersection_count_img, out_count_img)

        all_gt_count += gt_count_img
        all_out_count += out_count_img
        intersection_count += intersection_count_img

    recall = intersection_count / all_gt_count
    precision = intersection_count / all_out_count
    mAP = np.sum((intersection_class_out_count+1) / (class_out_count+1)) / 90
    return recall, precision, mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_in_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--folder_out_path", type=str, required=True)
    parser.add_argument("--folder_out_gt_path", type=str, required=True)
    parser.add_argument("--ground_truth_file", type=str, required=True)
    args = parser.parse_args()
    
    recall, precision, mAP = inference(args.folder_in_path, args.model_path, args.folder_out_path,args.folder_out_gt_path, args.ground_truth_file)

    print(f"recall:{recall}, precision:{precision}, map:{mAP}")


