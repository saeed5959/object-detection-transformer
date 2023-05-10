import json
import os

# opening file
dataset_folder_path = os.path.join(os.path.dirname(__file__),"dataset")
file_path = os.path.join(dataset_folder_path,"annotations","panoptic_val2017.json")
with open(file_path) as file:
    data_file_in = json.load(file)

# making data_file_out
annotations = data_file_in["annotations"] #list of dict : [{'segments_info', 'file_name', 'image_id'},...]
data_file_out = []
for data in annotations:
    data_line = "" 

    img_path = os.path.join(dataset_folder_path,"val2017",data["file_name"])
    data_line += img_path 

    for box in data["segments_info"]:
        category_id, bbox = box["category_id"], box["bbox"]
        data_line += "|" + str(category_id) + "," + str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3])
    
    data_file_out.append(data_line)
# saving file
with open(os.path.join(dataset_folder_path,"dataset_file.txt"), "w") as file:
    file.write(data_file_out[0]) 
    for line in data_file_out[1:]:
        file.write("\n" + line)


# print(data_file_in["annotations"][0].keys(),"\n")
# print(data_file_in["categories"][0].keys(),"\n")
# for category in data_file_in["categories"]:
#     print(category["name"],", ")