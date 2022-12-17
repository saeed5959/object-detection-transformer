import torch
from torch import nn
from torch.utils.data import DataLoader

from emotional_tts.models import TextToSpeech, Discriminator
from core.settings import train_config
from emotional_tts.data_utils import DatasetTTS


def main(training_files:str, model_path:str):
    
    train_dataset = DatasetTTS(training_files)
    
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
                              batch_size=train_config.batch_size)
    
    model_g = TextToSpeech().cuda()
    model_d = Discriminator().cuda()
    
    optim_g = torch.optim.AdamW(
        model_g.parameters(),
        train_config.learning_rate)
    
    optim_d = torch.optim.AdamW(
        model_d.parameters(),
        train_config.learning_rate)
    
    loss_d = nn.NLLLoss()
    loss_g = nn.NLLLoss()
    loss_duration = nn.NLLLoss()
    
    model_g.train()
    model_d.train()
    
    for epoch in range(train_config.epochs):
        for _, (x, x_length, y, y_length, speaker_id, emotion_id) in enumerate(train_loader):
            x, x_length = x.cuda(), x_length.cuda()
            y, y_length = y.cuda(), y_length.cuda()
            speaker_id = speaker_id.cuda()
            emotion_id = emotion_id.cuda()
            
            y_g, x_dur_mono, x_dur_pred = model_g(x, x_length, y, y_length, speaker_id, emotion_id)
            
            #not complete
            y_d, _ = model_d(y)
            
            loss_disc = loss_d(y_d,y_g)
            
            optim_d.zero_grad()
            loss_disc.backward()
            optim_d.step()
            
            loss_gen = loss_g.generator_loss(y,y_g)
            loss_dur = loss_duration(x_dur_mono, x_dur_pred)
            
            optim_g.zero_grad()
            loss_gen.backward()
            optim_g.step()

            print(f"===<< EPOCH : {epoch}  >>====")
            print(f'disc:{loss_disc} , gen{loss_gen}, dur:{loss_dur}')

        if epoch % train_config.save_model == 0:
            torch.save(model_g, model_path)

