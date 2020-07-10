import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import yaml


class Predictor():
    def __init__(self, config):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
        
        # Load vocabulary wrapper
        with open(config["vocab_path"], 'rb') as f:
            self.vocab = pickle.load(f)
    
        # Build models
        self.encoder = EncoderCNN(config["embed_size"]).eval()  # eval mode (batchnorm uses moving mean/variance)
        self.decoder = DecoderRNN(config["embed_size"], config["hidden_size"],
                                  len(self.vocab), config["num_layers"])
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
    
        # Load the trained model parameters
        self.encoder.load_state_dict(torch.load(config["encoder_path"]))
        self.decoder.load_state_dict(torch.load(config["decoder_path"]))
        
    def load_image(self, image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image
    
    def predict(self, image):
    
        # Prepare an image
        image = self.load_image(image, self.transform)
        image_tensor = image.to(self.device)
        
        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption[1:-1])
        
        return sentence
    
if __name__ == '__main__':
    with open("config.yaml", "rb") as rb:
        config = yaml.load(rb, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    model = Predictor(config)
    
    args = parser.parse_args()
    prediction = model.predict(args.image)
    print(prediction)