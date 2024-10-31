from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoProcessor
from underthesea import word_tokenize
import re
import json
import torch
import pandas as pd
from omegaconf import OmegaConf
from typing import List, Optional, Union
import transformers
from transformers.utils import TensorType
from PIL import Image
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ViVQADataset(Dataset):
    def __init__(self, dataframe_path, image_path, model_path = None, answers_path = "answer_path.json"):
        with open(answers_path, 'r') as f:
            self.vocab = json.loads(f.read())
            
        self.df = pd.read_csv(dataframe_path)
        self.df['answer2idx'] = self.df.apply(lambda x: self.vocab[x.answer], axis = 1)
  
        self.answers = list(self.df['answer2idx'])
        
        self.image_path = image_path
        if model_path == None:
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
        else:
            try:
                self.processor= AutoProcessor.from_pretrained(model_path)
            except:
                logger.info("Fail to load processor model!!!")
                

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df['question'].iloc[idx]
        
        img_id = self.df['img_id'].iloc[idx]
        image = Image.open(f'{self.image_path}/{img_id}.jpg')
        
        answer = self.answers[idx]
        
        #processor(text=texts, images=image, padding="max_length", return_tensors="pt")
        inputs = self.processor(text = question, images = image, 
                                return_tensors='pt', 
                                truncation=True,
                                padding='max_length')
        inputs |= {'labels': answer}
        return inputs

class ProcessedData:
    
    def __init__(self, train_df_path, test_df_path, train_image_path, test_image_path):
        self.train_df_path = train_df_path 
        self.test_df_path = test_df_path
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
    
    def processing(self):
        train_dataset = ViVQADataset(self.train_df_path, self.train_image_path)
        test_dataset = ViVQADataset(self.test_df_path, self.test_image_path)
        return train_dataset, test_dataset
    
if __name__ == '__main__':
    train_dataset, test_dataset = ProcessedData("csv_data/train.csv", "csv_data/test.csv", "images/train", "images/test").processing()