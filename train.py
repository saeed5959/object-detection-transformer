import argparse
from object_detection import train


def tts_train(train_file_path:str ,model_path:str):
    
    train.main(train_file_path, model_path)

    print(f"output model is saved in {model_path}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=False)
    args = parser.parse_args()
    
    tts_train(args.train_file_path,args.model_path)
                