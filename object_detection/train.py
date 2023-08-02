import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from object_detection import models
from core.settings import train_config, model_config
from object_detection.data_utils import DatasetObjectDetection, augmentation
from object_detection.utils import load_pretrained, noraml_weight

device = train_config.device

def main(training_files:str, model_path:str, pretrained: str):

    writer = SummaryWriter()    
    train_dataset = DatasetObjectDetection(training_files, augmentation)
    
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
                              batch_size=train_config.batch_size)

    model = models.VitModel().to(device)

    if pretrained != "":
        model, step_all, epo, lr = load_pretrained(model, pretrained, device)
        print("pretrained model loaded!")

    else:
        step_all = 0
        epo = torch.tensor([0]).to(device)
        lr = train_config.learning_rate
        print("pretrained model didn't load!")

    optim = torch.optim.AdamW(model.parameters(), lr)
    lr_schedular = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=train_config.lr_end / lr, total_iters=40)
    #combination of sigmoid and nll loss for 
    loss_obj = nn.BCEWithLogitsLoss(reduction="none")
    #combination of softmax and nll loss
    class_weight = noraml_weight(model_config.panoptic_file_path).to(device)
    print(class_weight)
    loss_class = nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    #we use "sum" instead of "mean" : because of mask
    loss_box = nn.L1Loss(reduction="none")    
    #loss poa
    loss_poa = nn.BCELoss(reduction="none")

    model.train()
    


    for epoch in range(1, train_config.epochs+1):
        epo += 1
        print(f"===<< EPOCH : {epoch}  >>====")
        loss_obj_all = 0
        loss_class_all = 0
        loss_bbox_all = 0
        loss_poa_all = 0

        for step, (img, obj_id, class_input, bbox_input, poa_input, mask_obj, mask_class, mask_bbox, mask_poa) in enumerate(train_loader):
            img, bbox_input, class_input, obj_id, poa_input = img.to(device), bbox_input.to(device), class_input.to(device), obj_id.to(device), poa_input.to(device)
            mask_obj, mask_class, mask_bbox, mask_poa = mask_obj.to(device), mask_class.to(device), mask_bbox.to(device), mask_poa.to(device)
            
            out, similarity_matrix = model(img, poa_input, epo)

            #loss object
            loss_obj_out = torch.sum(loss_obj(out[:,:,0], obj_id) * mask_obj)
            #normalize
            loss_obj_out = (loss_obj_out / (torch.tensor([train_config.batch_size]).to(device) * torch.tensor([256]).to(device))).squeeze(-1)

            #loss class with mask
            class_out = out[:,:,1:model_config.class_num+1]
            loss_class_out = torch.sum(loss_class(class_out.transpose(1,2), class_input.transpose(1,2)) * mask_class)
            #normalize
            loss_class_out = loss_class_out / torch.sum(obj_id)
            
            #loss bbox with mask
            box_out = out[:,:,model_config.class_num+1:]
            #bounding box between [0,1]
            box_out = torch.minimum(torch.tensor([1]).to(device), torch.maximum(torch.tensor([0]).to(device), box_out))
            loss_box_out = torch.sum(loss_box(box_out, bbox_input) * mask_bbox)
            #normalize
            loss_box_out = loss_box_out / torch.sum(obj_id)
            
            #loss poa
            loss_poa_out = torch.sum(loss_poa(similarity_matrix, poa_input) * mask_poa)
            #normalize
            loss_poa_out = (loss_poa_out / (torch.sum(obj_id) * torch.tensor([256]).to(device))).squeeze(-1)

            loss_all = loss_obj_out + loss_box_out + loss_class_out + loss_poa_out
        
            optim.zero_grad()
            loss_all.backward()
            if train_config.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_norm)
            optim.step()
            
            loss_obj_all += loss_obj_out
            loss_class_all += loss_class_out
            loss_bbox_all += loss_box_out
            loss_poa_all += loss_poa_out

            if step % train_config.step_show == 0 and step!=0:
                step_all += train_config.step_show
                #writing in tensorboard
                loss_obj_all = loss_obj_all / train_config.step_show
                loss_class_all = loss_class_all / train_config.step_show
                loss_bbox_all = loss_bbox_all / train_config.step_show
                loss_poa_all = loss_poa_all / train_config.step_show

                writer.add_scalar("loss_obj", loss_obj_all, step_all)
                writer.add_scalar("loss_class", loss_class_all, step_all)
                writer.add_scalar("loss_box", loss_bbox_all, step_all)
                writer.add_scalar("loss_poa", loss_poa_all, step_all)

                #printing loss
                print(f"===<< STEP : {step}  >>====")
                print(f'loss_obj : {loss_obj_all} , loss_class : {loss_class_all} , loss_box : {loss_bbox_all} , loss_poa : {loss_poa_all}')

                loss_obj_all = 0
                loss_class_all = 0
                loss_bbox_all = 0
                loss_poa_all = 0

        lr_schedular.step()
        print(f"lr = {lr_schedular.get_last_lr()}")
        
        if epoch % train_config.save_model == 0:
            torch.save({'model':model.state_dict(),
                        'step_all':step_all,
                        'epoch':epo,
                        'lr':lr_schedular.get_last_lr()}, model_path)

