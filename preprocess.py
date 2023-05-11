import json
import os
import cv2
import numpy as np


def img_preprocess(img_path: str, img_path_out: str):

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img = cv2.resize(img,(256,256))
    cv2.imwrite(img_path_out, img)

    return height, width

def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

def normalize_bbox(bbox: list, height: int, width: int):

    c0_x, c0_y, w, h = bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height
    c_x = c0_x + w/2, 
    c_y = c0_y + h/2

    return c_x, c_y, w, h

def color_object_segment(segment_path: str, data: list):
    #consider rgb or gray color
    data_color = []
    segment = cv2.imread(segment_path,0)
    for box in data:
        category_id, bbox = box["category_id"], box["bbox"]#c0_x, c0_y, w, h = bbox
        c0_x, c0_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        segment_box = segment[c0_y:c0_y+h ,c0_x:c0_x+w]

        unique_values = np.unique(segment_box)
        counts = np.array([np.count_nonzero(segment_box == val) for val in unique_values])
        max_index = np.argmax(counts)
        most_repetitive_value = unique_values[max_index]
        data_color.append(most_repetitive_value)


def main():

    dataset_folder_path = os.path.join(os.path.dirname(__file__),"dataset")
    file_path = os.path.join(dataset_folder_path,"annotations","panoptic_val2017.json")
    segment_folder_path =  os.path.join(dataset_folder_path,"annotations","val2017")
    img_folder_path =  os.path.join(dataset_folder_path,"panoptic_val2017")
    img_folder_path_out = os.path.join(dataset_folder_path,"val2017_out")

    with open(file_path) as file:
        data_file_in = json.load(file)

    # making data_file_out
    annotations = data_file_in["annotations"] #list of dict : [{'segments_info', 'file_name', 'image_id'},...]
    data_file_out = []
    for data in annotations:
        data_line = "" 

        img_path = os.path.join(img_folder_path, data["file_name"][:-3]+"jpg")
        segment_path = os.path.join(segment_folder_path, data["file_name"])
        img_path_out = os.path.join(img_folder_path_out, data["file_name"][:-3]+"jpg")
        height, width = img_preprocess(img_path, img_path_out)
        data_line += img_path_out 

        
        color_object_segment(segment_path, data["segments_info"])
        for box in data["segments_info"]:
            category_id, bbox = box["category_id"], box["bbox"]#c0_x, c0_y, w, h = bbox
            c0_x, c0_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            segment[]


            c_x, c_y, w, h = normalize_bbox(bbox, height, width)
            data_line += "|" + str(category_id) + "," + str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3])
        
        data_file_out.append(data_line)


    file_path_out = os.path.join(dataset_folder_path,"dataset_file.txt")
    save_file(file_path_out, data_file_out)


# print(data_file_in["annotations"][0].keys(),"\n")
# print(data_file_in["categories"][0].keys(),"\n")
# for category in data_file_in["categories"]:
#     print(category["name"],", ")