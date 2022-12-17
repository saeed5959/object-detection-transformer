# Emotional TTS With Variational Auto Encoder with flow-based block and with Gan generator for high quality and emotional speech

## Converting Text To Speech With Emotion for Multi Speaker

## Docker FILe is also available

## HOw To Run
  Train
  
  
    python3 main.py --model train   --train_file_path  ./train_file.txt   --model_path  ./gen.pth
  
  Inference
  
  
    python3 main.py --model inference   --text  "how are you"   --model_path  ./gen.pth   --emotion 1   --voice_path  ./test.wav
    
    
 
 ## This neural net consists of 5 blocks :
 
 
    1- Text Encoder : based on Transformer
    2- Duration Predictor : based on Convolution_1D block
    3- Flow Block : based on flow net
    4- Decoder : based on Generator and Discriminator 
    5- whole network : based on VAE
  
 ## EMOTION
  Adding emotion embedding layer in text encoder and flow based block
  
 ## Multi Speaker
 Adding speaker embedding layer in text encoder and flow based block
