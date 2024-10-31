from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoProcessor
import re
import json
import torch
import pandas as pd
from typing import List, Optional, Union
import transformers
from transformers.utils import TensorType
from PIL import Image
# import os
# cert = r"Zscaler Root CA.crt"
# os.environ["REQUESTS_CA_BUNDLE"] = cert

import logging

logging.basicConfig(level=logging.DEBUG)

class ViVQADataset(Dataset):
    def __init__(self, dataframe_path, image_path, model_path = None, answers_path = "answer_path.json"):
        
        logging.info("Loading answer vocab")
        with open(answers_path, 'r') as f:
            self.vocab = json.loads(f.read())
            
        self.df = pd.read_csv(dataframe_path)
        self.df['answer2idx'] = self.df.apply(lambda x: self.vocab[x.answer], axis = 1)
  
        self.answers_ids = list(self.df['answer2idx'])
        self.answers = list(self.df['answer'])
        
        self.image_path = image_path
        if model_path == None:
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
        else:
            try:
                self.processor= AutoProcessor.from_pretrained(model_path)
            except:
                logging.info("Fail to load processor model!!!")
                

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df['question'].iloc[idx]
        
        img_id = self.df['img_id'].iloc[idx]
        image = Image.open(f'{self.image_path}/{img_id}.jpg')
        
        answer = self.answers[idx]
        
        #processor(text=texts, images=image, padding="max_length", return_tensors="pt")
        inputs = self.processor(text = question + answer, images = image, 
                                return_tensors='pt', 
                                truncation=True,
                                padding='max_length')
        # inputs |= {'labels': answer}
        inputs = {"input_ids": inputs['input_ids'][0], "pixel_values": inputs['pixel_values'][0]}
        attention_mask = 1 - inputs['input_ids']
        attention_mask[attention_mask <0] = 1
        inputs |= {"attention_mask" : attention_mask}
        # print(inputs)
        return inputs

class ProcessedData:
    
    def __init__(self, train_df_path, test_df_path, train_image_path, test_image_path):
        self.train_df_path = train_df_path 
        self.test_df_path = test_df_path
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
    
    def processing(self):
        logging.info("Processing train data!")
        train_dataset = ViVQADataset(self.train_df_path, self.train_image_path)
        
        logging.info("Processing test data!")
        test_dataset = ViVQADataset(self.test_df_path, self.test_image_path)
        return train_dataset, test_dataset
    
if __name__ == '__main__':
    train_dataset, test_dataset = ProcessedData("csv_data/train.csv", "csv_data/test.csv", "images/train", "images/test").processing()
    for d in train_dataset:
        print(d['input_ids'].size())
        print(d['pixel_values'].size())
        break