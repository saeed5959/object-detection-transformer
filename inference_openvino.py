import torch
import argparse
from torch.nn.functional import sigmoid, softmax
import onnx
from openvino.runtime import Core

from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img, show_box
from core.settings import train_config,model_config

device = train_config.device


def convert_onnx(model, onnx_path, input_img):
	
    torch.onnx.export(model, input_img, onnx_path, input_names=['input'], output_names=['output'], export_params=True)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return 

        
def convert_openvino(onnx_path):

    core = Core()
    model_onnx = core.read_model(model=onnx_path)
    model_openvino = core.compile_model(model=model_onnx, device_name="AUTO")

    return model_openvino
        

def inference_test(img_path : str, model_path : str):
    
    #load model
    model = models.VitModel().to(device)
    model, step_all, epo, lr = load_pretrained(model, model_path, device)
    model.eval()

    #prepare input image
    img = img_preprocess_inference(img_path)
    img = img.to(device)
    
    poa = []
    epoch = torch.tensor([30]).to(device)

    #for tensorrt : first convert to onnx then convert onnx to tensorrt
    onnx_path = "./model_onnx.onnx"
    # convert_onnx(model, onnx_path, img)
    model_openvino = convert_openvino(onnx_path)

    out = model_openvino([img.numpy()])[0]

    out = torch.tensor(out)
    #postprocessing - sigmoid for object
    obj_out = sigmoid(out[:,:,0])
    #softmax for class
    class_out = softmax(out[:,:,1:model_config.class_num+1], dim=-1)
    #bound [0,1] for bbox
    box_out = out[:,:,model_config.class_num+1:]
    box_out = torch.minimum(torch.tensor([1]).to(device), torch.maximum(torch.tensor([0]).to(device), box_out.to(device)))

    return obj_out[0].detach().cpu().numpy(), class_out[0].detach().cpu().numpy(), box_out[0].detach().cpu().numpy()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    

    obj_out, class_out, box_out = inference_test(args.img_path, args.model_path)

    obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final = nms_img(obj_out, class_out, box_out)
    print(obj_score_list_final, class_list_final, class_score_list_final, box_list_final)

    show_box(args.img_path, class_list_final, box_list_final, args.out_path)
