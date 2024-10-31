from transformers import AutoProcessor, AutoModel
import torch.nn as nn
from PIL import Image
import os
# cert = r"Zscaler Root CA.crt"
# os.environ["REQUESTS_CA_BUNDLE"] = cert

class CustomModel(nn.Module):
    
    def __init__(self, model_name = "google/siglip-base-patch16-256-multilingual"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, pixel_values, attention_mask, return_loss = True, return_dict = True):
        if return_loss == True:
            return self.model(input_ids = input_ids, pixel_values = pixel_values, attention_mask = attention_mask, return_loss = True, return_dict = return_dict)
        else:
            return self.model(input_ids = input_ids, pixel_values = pixel_values, attention_mask = attention_mask, return_loss = False, return_dict = return_dict)

if __name__ == "__main__":
    url = r"images\test\322.jpg"
    image = Image.open(url)
    texts = "Bệnh nhân tên gì?"
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
    inputs = processor(text=texts, images=image, padding="max_length", truncation = True, return_tensors="pt")
    attention_mask = 1 - inputs['input_ids']
    attention_mask[attention_mask <0] = 1
    inputs |= {"attention_mask": attention_mask}
    print(inputs['input_ids'].size(), inputs['pixel_values'].size(), inputs['attention_mask'].size())
    
    model = CustomModel()
    print(model(inputs['input_ids'], inputs['pixel_values']), inputs['attention_mask'])
    
    