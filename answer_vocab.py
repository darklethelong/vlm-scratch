import pandas as pd
from collections import defaultdict
import os
import json

class ProcessingAnswer:
    
    def __init__(self, train_csv_path, test_csv_path):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.vocab_answer = defaultdict(lambda : len(self.vocab_answer))
        
    def processing(self):
        if os.path.exists("answer_path.json"):  
            with open("answer_path.json", 'r') as f:
                self.vocab_answer = json.loads(f.read())
            return self.vocab_answer
        else:
            train_df = pd.read_csv(self.train_csv_path)
            test_df = pd.read_csv(self.test_csv_path)
            
            answers = list(set(list(train_df['answer']) + list(test_df['answer'])))
            for answer in answers:
                self.vocab_answer[answer]
            with open("answer_path.json", "w") as f:
                json.dump(self.vocab_answer, f)
            return self.vocab_answer
            
if __name__ == "__main__":
    vocab_answer = ProcessingAnswer("csv_data/train.csv", "csv_data/test.csv").processing()
    print(vocab_answer)
        