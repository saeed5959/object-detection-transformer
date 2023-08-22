import torch
import argparse
from torch.nn.functional import sigmoid, softmax
import onnx
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

from object_detection import models
from object_detection.utils import load_pretrained, img_preprocess_inference, nms_img, show_box
from core.settings import train_config,model_config

device = train_config.device
TRT_LOGGER = trt.Logger()


def convert_onnx(model, onnx_path, input_img):
	
    torch.onnx.export(model, input_img, onnx_path, input_names=['input'], output_names=['output'], export_params=True)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return 

    
        
def convert_trt(model_file, trt_path, max_ws=512*1024*1024, fp16=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    # builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            engine = builder.build_engine(network, config=config)
            if engine is None:
                raise RuntimeError("Fail to build the trt model")
            
            #save trt model
            with open(trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))


            return engine
        

def load_trt(trt_path):

    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, 'rb') as f:
        engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

        return engine
    

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
    trt_path = "./model_trt.trt"
    convert_onnx(model, onnx_path, img)
    engine = convert_trt(onnx_path, trt_path)
    #optional
    engine = load_trt(trt_path)
    context = engine.create_execution_context()
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    host_input = np.array(img.cpu().numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    out = torch.Tensor(host_output).reshape(1, output_shape[1], output_shape[2])

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
    
    t1= time.time()
    obj_out, class_out, box_out = inference_test(args.img_path, args.model_path)
    t2 = time.time()
    print(t2-t1)

    obj_score_list_final, class_list_final, class_score_list_final, box_list_final, xy_list_final = nms_img(obj_out, class_out, box_out)
    print(obj_score_list_final, class_list_final, class_score_list_final, box_list_final)

    show_box(args.img_path, class_list_final, box_list_final, args.out_path)
