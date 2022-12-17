import argparse
from emotional_tts import train, inference


def tts_train(train_file_path:str ,model_path:str):
    
    train.main(train_file_path, model_path)

    print(f"output model is saved in {model_path}")


def tts_infer(text:str, model_path:str, emotion:str, voice_path:str):
    
    inference.main(text, model_path, emotion, voice_path)
    
    print(f"generated voice is saved in {voice_path}")
    
    return 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--train_file_path", type=str, required=False)
    parser.add_argument("--text", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--emotion", type=str, required=False)
    parser.add_argument("--voice_path", type=str, required=False)
    args = parser.parse_args()
    
    if args.mode == "train":
        tts_train(args.train_file_path,args.model_path)
        
    elif args.mode == "inference":
        tts_infer(args.text, args.model_path, args.emotion, args.voice_path)
        
    else:
        print("please enter your --mode from these : train  , inference ")
                