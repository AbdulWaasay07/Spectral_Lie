from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os

def download():
    model_name = "facebook/wav2vec2-base"
    print(f"Pre-downloading {model_name}...")
    Wav2Vec2Processor.from_pretrained(model_name)
    Wav2Vec2Model.from_pretrained(model_name)
    print("Download complete.")

if __name__ == "__main__":
    download()
