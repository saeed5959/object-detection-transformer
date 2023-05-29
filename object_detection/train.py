import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from object_detection import models
from core.settings import train_config, model_config
from object_detection.data_utils import DatasetObjectDetection, augmentation
from object_detection.utils import load_pretrained

device = train_config.device

def main(training_files:str, model_path:str, pretrained: str):

    writer = SummaryWriter()    
    train_dataset = DatasetObjectDetection(training_files, augmentation)
    
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
                              batch_size=train_config.batch_size)

    model = models.VitModel().to(device)
    if pretrained != "":
        model = load_pretrained(model, pretrained, device)
        print("pretrained model loaded!")
    else:
        print("pretrained model didn't load!")

    optim = torch.optim.AdamW(model.parameters(), train_config.learning_rate)
    #combination of sigmoid and nll loss for 
    loss_obj = nn.BCEWithLogitsLoss()
    #combination of softmax and nll loss
    loss_class = nn.CrossEntropyLoss(reduction="sum")
    #we use "sum" instead of "mean" : because of mask
    loss_box = nn.MSELoss(reduction="sum")    
    model.train()
    
    step_all = 0
    for epoch in range(1, train_config.epochs+1):
        print(f"===<< EPOCH : {epoch}  >>====")
        loss_obj_all = 0
        loss_class_all = 0
        loss_bbox_all = 0

        for step, (img, obj_id, class_input, bbox_input, mask_class, mask_bbox) in enumerate(train_loader):
            img, bbox_input, class_input, obj_id = img.to(device), bbox_input.to(device), class_input.to(device), obj_id.to(device)
            mask_class, mask_bbox = mask_class.to(device), mask_bbox.to(device)
            
            out = model(img)
            
            #loss object
            loss_obj_out = loss_obj(out[:,:,0], obj_id)

            #loss class with mask
            class_out = out[:,:,1:model_config.class_num+1]*mask_class
            loss_class_out = loss_class(class_out, class_input*mask_class)
            #normalize
            loss_class_out = loss_class_out / torch.sum(mask_class) * model_config.class_num
            
            #loss bbox with mask
            box_out = out[:,:,model_config.class_num+1:]*mask_bbox
            #bounding box between [0,1]
            box_out = torch.minimum(torch.Tensor([1]), torch.maximum(torch.Tensor([0]), box_out))
            loss_box_out = loss_box(box_out, bbox_input*mask_bbox)
            #normalize
            loss_box_out = loss_box_out / torch.sum(mask_bbox) * 4
            

            loss_all = loss_obj_out + loss_box_out + loss_class_out
        
            optim.zero_grad()
            loss_all.backward()
            optim.step()
            
            loss_obj_all += loss_obj_out
            loss_class_all += loss_class_out
            loss_bbox_all += loss_box_out

            if step % train_config.step_show == 0:
                step_all += step
                #writing in tensorboard
                loss_obj_all = loss_obj_all / train_config.step_show
                loss_class_all = loss_class_all / train_config.step_show
                loss_bbox_all = loss_bbox_all / train_config.step_show

                writer.add_scalar("loss_obj", loss_obj_all, step_all)
                writer.add_scalar("loss_class", loss_class_all, step_all)
                writer.add_scalar("loss_box", loss_bbox_all, step_all)

                #printing loss
                print(f"===<< STEP : {step}  >>====")
                print(f'loss_obj : {loss_obj_out} , loss_class : {loss_class_out} , loss_box : {loss_box_out}')

                loss_obj_all = 0
                loss_class_all = 0
                loss_bbox_all = 0

        if epoch % train_config.save_model == 0:
            torch.save(model.state_dict(), model_path)

