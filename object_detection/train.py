import torch
from torch import nn
from torch.utils.data import DataLoader

from object_detection import models
from core.settings import train_config
from object_detection.data_utils import DatasetObjectDetection, augmentation


def main(training_files:str, model_path:str, device: str):
    
    train_dataset = DatasetObjectDetection(training_files, augmentation)
    
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
                              batch_size=train_config.batch_size)

    model = models.VitModel()
    optim = torch.optim.AdamW(model.parameters(), train_config.learning_rate)
    loss_class = nn.CrossEntropyLoss()
    loss_box = nn.MSELoss()    
    model.train()
    model.to(device)
    
    for epoch in range(train_config.epochs):
        print(f"===<< EPOCH : {epoch}  >>====")
        for step, (img, class_input, bbox_input, class_mask) in enumerate(train_loader):
            img, bbox_input, class_input, class_mask = img.to(device), bbox_input.to(device), class_input.to(device), class_mask.to(device)
            
            class_out, bbox_out = model(img)
            
            loss_class_out = loss_class(class_out, class_input)

            #ignore background
            bbox_input_mask = torch.masked_select(bbox_input, class_mask>0)
            bbox_output_mask = torch.masked_select(bbox_out, class_mask>0)
            loss_box_out = loss_box(bbox_input_mask, bbox_output_mask)
            
            loss_all = loss_box_out + loss_class_out
        
            optim.zero_grad()
            loss_all.backward()
            optim.step()
            
            print(f"===<< STEP : {step}  >>====")
            print(f'loss_class : {loss_class_out} , loss_box : {loss_box_out}')

        if epoch % train_config.save_model == 0:
            torch.save(model, model_path)

