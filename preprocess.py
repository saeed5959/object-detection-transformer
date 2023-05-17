import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from core.settings import model_config

def img_preprocess(img_path: str, img_path_out: str):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    cv2.imwrite(img_path_out, img)

    return

def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

    return

def normalize_bbox(bbox: list, height: int, width: int):

    c0_x, c0_y, w, h = bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height
    c_x = c0_x + w/2
    c_y = c0_y + h/2

    return c_x, c_y, w, h

def find_most_repetitive(data):

    most_repetitive_value = []
    for rgb in range(3):
        unique_values, counts = np.unique(data[:,:,rgb], return_counts=True)
        max_index = np.argmax(counts)
        most_repetitive_value.append(unique_values[max_index])

    return most_repetitive_value


def color_object_segment(segment_path: str, data: list):

    data_list = []
    color_list = []
    segment = cv2.imread(segment_path)
    for box in data:
        if box["category_id"] < model_config.class_num:
            bbox = box["bbox"]
            c0_x, c0_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            segment_box = segment[c0_y:c0_y+h ,c0_x:c0_x+w]
            
            color = find_most_repetitive(segment_box)
            color_list.append(color)
            data_list.append({"category_id":box["category_id"], "bbox":box["bbox"]})

    return data_list, color_list


def patch_info(img_path: str, segment_path: str, data_list: list, color_list: list):

    patch_num_h = model_config.patch_num_h
    segment = cv2.imread(segment_path)
    height, width, _ = segment.shape
    step_size_height = height/patch_num_h
    step_size_width = width/patch_num_h

    background_id = 0
    patch_data = img_path
    for h in range(patch_num_h):
        for w in range(patch_num_h):
            segment_box = segment[int(h*step_size_height):int((h+1)*step_size_height),int(w*step_size_width):int((w+1)*step_size_width)]
            color = find_most_repetitive(segment_box)

            if color in color_list:
                color_index = color_list.index(color)
                category_id, bbox = data_list[color_index]["category_id"], data_list[color_index]["bbox"]
                c_x, c_y, w, h = normalize_bbox(bbox, height, width)
                patch_data += "|" + str(category_id) + "," + str(c_x) + "," + str(c_y) + "," + str(w) + "," + str(h)

            else:
                patch_data += "|" + str(background_id) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0)


    return patch_data


def main():

    dataset_folder_path = os.path.join(os.path.dirname(__file__),"dataset")
    file_path = os.path.join(dataset_folder_path,"annotations","panoptic_val2017.json")
    segment_folder_path =  os.path.join(dataset_folder_path,"annotations","panoptic_val2017")
    img_folder_path =  os.path.join(dataset_folder_path,"val2017")
    img_folder_path_out = os.path.join(dataset_folder_path,"val2017_out")

    with open(file_path) as file:
        data_file_in = json.load(file)

    # list of dict : [{'segments_info', 'file_name', 'image_id'},...]
    annotations = data_file_in["annotations"] 
    data_file_out = []
    for data in tqdm(annotations):

        img_path = os.path.join(img_folder_path, data["file_name"][:-3]+"jpg")
        segment_path = os.path.join(segment_folder_path, data["file_name"])
        img_path_out = os.path.join(img_folder_path_out, data["file_name"][:-3]+"jpg")
        img_preprocess(img_path, img_path_out)

        data_list, color_list = color_object_segment(segment_path, data["segments_info"])
        patch_data = patch_info(img_path_out, segment_path, data_list, color_list)
        data_file_out.append(patch_data)

    file_path_out = os.path.join(dataset_folder_path,"dataset_file_out.txt")
    save_file(file_path_out, data_file_out)

    return

main()
