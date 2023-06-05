import argparse
import numpy as np
import cv2
import torch

from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img, show_box
from core.settings import train_config

device = train_config.device

def inference_test(img_path : str, model_path : str):

    #load model
    model = models.VitModel().to(device)
    model = load_pretrained(model, model_path, device)
    model.eval()

    #prepare input image
    img = img_preprocess_inference(img_path)
    img.to(device)

    poa = []
    epoch = torch.tensor([30]).to(device)
    #giving input to model
    obj_out, class_out, box_out = model.inference(img, poa, epoch)

    return obj_out[0].detach().numpy(), class_out[0].detach().numpy(), box_out[0].detach().numpy()

def show_obj(obj_out, class_out, box_out):
    img = np.ones((16,16,3))*255
    for patch in range(len(obj_out)):
        if obj_out[patch] > 0.4:
            x = patch % 16
            y = patch // 16
            class_id = np.argmax(class_out[patch])
            print(class_id)
            print(box_out[patch])

            img[int(y),int(x),:] = [class_id, class_id, class_id]

    cv2.imwrite("./x_872.jpg",img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    
    # print(inference_test(args.img_path, args.model_path))
    obj_out, class_out, box_out = inference_test(args.img_path, args.model_path)
    print(obj_out, class_out, box_out)
    show_obj(obj_out, class_out, box_out)

    obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final = nms_img(obj_out, class_out, box_out)
    print(obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final)

    show_box(args.img_path, class_list_final, box_list_final, args.out_path)

    #apply overlap for same category
