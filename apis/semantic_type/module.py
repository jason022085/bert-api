import torch
import os
import re
from os.path import join
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# 載入 pytorch 與 Bert 相關套件
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW

import pandas as pd
import numpy as np


valid_types = np.array(('CARDINAL', 'CITY_COUNTY', 'COUNTRY', 'DATE', 'DISTRICT_TOWN', 'EVENT', 'FAC', 'GPE',
                'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT',
                'QUANTITY', 'TIME', 'WORK_OF_ART', 'address', 'affiliation', 'age', 'agency', 'album', 'area', 'artist', 'assets', 'borough', 'brand',
                'capacity', 'category', 'city', 'class', 'classification', 'club', 'code', 'collection', 'command', 'company', 'component', 'continent',
                'country', 'county', 'creator', 'credit', 'critical', 'currency',
                'date', 'day', 'depth', 'description', 'director', 'duration', 'education', 'elevation', 'email', 'family', 'format',
                'gender', 'genre', 'grades', 'industry', 'isbn', 'isin', 'isrc', 'iswc', 'language', 'latitude', 'location', 'longitude',
                'manufacturer', 'marketValue', 'name', 'nationality', 'notes', 'operator', 'order', 'organization', 'origin', 'owner',
                'performanceIndicator', 'person', 'phone', 'plays', 'position', 'product', 'profits', 'publisher',
                'range', 'rank', 'ranking', 'region', 'religion', 'requirement', 'resourceIndicators', 'result', 'revenue',
                'sales', 'service', 'sex', 'species', 'state', 'status', 'street', 'symbol',
                'team', 'teamName', 'type', 'url', 'weight', 'year', 'zip', 'zipCode', 'other'))  # 115 + "Unsupported"
valid_index = list(range(len(valid_types)))
valid_dict = dict(zip(valid_index, valid_types))

pre_trained_loc = '.\\pretrained_bert'
device = "cpu"

# 載入 Bert 套件與 tokenizer
tokenizer = BertTokenizer.from_pretrained(os.path.join(pre_trained_loc,'./vocab.txt'), do_lower_case=True) # 'bert-base-multilingual-cased'

# 自定義 Bert 分類器函數


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # 指定 BERT 輸入長度大小(D_in), 分類器的隱藏層大小(H), 以及分類目標值的種類數量(D_out)
        D_in, H, D_out = 768, 128, 115
        # 載入 Bert 預訓練權重作為初始值
        self.bert = BertModel.from_pretrained(pre_trained_loc) #'bert-base-multilingual-cased'

        # 初始化自定義分類器的類神經網路
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        # 凍結 Bert 部分的權重
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 將資料輸入 BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # 將輸出結果存在 last_hidden_state_cls 中
        # last hidden state:模型最後一層輸出的隱藏狀態 shape: batch size,sequence length, 768
        last_hidden_state_cls = outputs[0][:, 0, :]
        # 將輸出結果輸入自定義分類器
        logits = self.classifier(last_hidden_state_cls)

        return logits


# 初始化 Bert 分類器
bert_classifier = BertClassifier(freeze_bert=False)
# 告訴 PyTorch 模型需要在 GPU/CPU 上執行
bert_classifier.to(device)
optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)
# 讀取模型參數
bert_classifier.load_state_dict(torch.load(os.path.join(
    pre_trained_loc, "model_weight.pt"), map_location='cpu'))
# 評價模式
bert_classifier.eval()


def column_to_text(dataframe):
    column_list = []
    cols = list(dataframe.columns)
    for c in cols:
        column_value = list(dataframe[c])
        column_text = " ".join([str(_) for _ in column_value])
        column_list.append(column_text)
    return column_list

# 設定 Bert 的前處理函數
def preprocessing_for_bert(data):
    data = [d[:128] for d in data]
    # 初始化要傳回的資料
    input_ids = []
    attention_masks = []
    # 把所有文句用 tokenizer 編碼
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 套用簡化版前處理函數
            add_special_tokens=True,        # 加上 `[CLS]` 與 `[SEP]`
            max_length=128,                 # 需要填充的最大長度
            pad_to_max_length=True,         # 是否要填充到最大長度
            return_attention_mask=True      # 是否傳回 attention mask
        )
        # 更新要傳回的資料
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # 將傳回資料轉為 tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

class Prediction(object):
    @staticmethod
    def evaluate(df, threshold=0.9, k=10):
        X_new = column_to_text(df)
        new_inputs, new_masks = preprocessing_for_bert(X_new)
        new_inputs.to(device)
        new_masks.to(device)
        with torch.no_grad():
            logits = bert_classifier(new_inputs, new_masks)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.array([p.argsort()[::-1][:k] for p in probs]) # 輸出機率前k大的類別
        # preds = [np.argmax(p) if np.max(p) >= threshold else 115 for p in probs] # 輸出機率最大且有過閥值的類別
        # preds = [np.argmax(p) for p in probs] # 輸出機率最大的類別
        return {
                'result': [[valid_dict[_] for _ in p if _ >= threshold] for p in preds],
                'message': "success"
            }