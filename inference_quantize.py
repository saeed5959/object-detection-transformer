import torch
from torch.ao.quantization import QConfigMapping ,get_default_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx
import argparse
from torch.nn.functional import sigmoid, softmax
from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img, show_box
from core.settings import train_config,model_config

#add this to avoid error for torch.fx for quantization for rearrange
from einops import rearrange
torch.fx.wrap('rearrange')

device = train_config.device


def quantize(model, input, mode="dynamic"):

    if mode=="dynamic":
        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, input)
        model_quantized = quantize_fx.convert_fx(model_prepared)

    if mode=="static":
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, input)
        model_quantized = quantize_fx.convert_fx(model_prepared)

    return model_quantized


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

    model_q = quantize(model, img)

    out, similarity_matrix = model_q(img, poa, epoch)

    #postprocessing - sigmoid for object
    obj_out = sigmoid(out[:,:,0])
    #softmax for class
    class_out = softmax(out[:,:,1:model_config.class_num+1], dim=-1)
    #bound [0,1] for bbox
    box_out = out[:,:,model_config.class_num+1:]
    box_out = torch.minimum(torch.tensor([1]).to(device), torch.maximum(torch.tensor([0]).to(device), box_out))

    torch.save({'model':model_q.state_dict(),
                        'step_all':step_all,
                        'epoch':epo,
                        'lr':1}, "./x_56_q_static.pth")

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

