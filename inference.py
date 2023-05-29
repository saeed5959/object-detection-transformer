import argparse

from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img
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

    #giving input to model
    obj_out, class_out, box_out = model.inference(img)

    #out_nms = nms_img(out)


    return obj_out, class_out, box_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    print(inference_test(args.img_path, args.model_path))
