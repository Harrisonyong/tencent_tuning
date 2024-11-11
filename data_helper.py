#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_helper.py
@Time    :   2024/05/20 15:37:32
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   dataprocess,将数据转为input_batch  
'''

import os
import json
import random
import torch
import csv
import transformers
from typing import List, Dict
from openpyxl import Workbook
import pandas as pd
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer
from config import Config

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DataSplit:
    """
        divide dataset to train and eval
    """
    data_path = Config.data_path
    val_set_size = Config.val_set_size
    
    @classmethod
    def load_data(cls):
        with open(cls.data_path, "r") as f:
            data = json.load(f)
        return data
    
    @classmethod
    def gen_data(cls):
        data = cls.load_data()
        random.shuffle(data)

        train_data = data[cls.val_set_size:]
        valid_data = data[:cls.val_set_size]
        
        return train_data, valid_data


class TencentDataset(Dataset):
    """
        dataset for supervised fine_tuning
    """
    def __init__(self, raw_data:List, tokenizer:transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = Config.sequence_len
        data_dict = self.preprocess(raw_data)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i)-> Dict[str, torch.Tensor]:
        return dict(
            input_ids = self.input_ids[i], 
            labels = self.labels[i],
            attention_mask = self.attention_mask[i]
            )

    def preprocess(self,raw_data:List[dict]):
        """Preprocesses the data for supervised fine-tuning."""
        raw_texts = []
        
        for single_data in raw_data:
            input = single_data["input"]
            output = single_data["output"]
            tokenized_data = self.tokenizer.encode("你是一个由腾讯开发的有用的人工智能助手。<unused5>\n") + self.tokenizer.encode(input) + [0] + self.tokenizer.encode(output) + [20] + self.tokenizer.encode("\n")
            tokenized_data = tokenized_data + [2] * (self.max_len - len(tokenized_data))            
            raw_texts.append(tokenized_data)
        
        input_ids = torch.tensor(raw_texts, dtype=torch.int)
        target_ids = input_ids.clone()
        target_ids[target_ids == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids = input_ids,
            labels = target_ids,
            attention_mask = attention_mask
        )
            

class CSVTransfer:
    """
        解析csv文件中的问答对，将符合最大长度的问答组合为json训练输入文件，不满足的返回为xlsx进行重新总结
    """
    def __init__(self,csv_path, tokenizer, out_csv, train_json) -> None:
        self.csv_path = csv_path
        self.tokenzier = tokenizer
        self.out_csv = out_csv
        self.train_json = train_json
    
    def read_and_recompose(self):
        exced_lines = []
        train_datas = []
        with open(self.csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                input = row[0]
                output = row[1]
                input_ids = self.tokenzier(input).input_ids
                output_ids = self.tokenzier(output).input_ids
                inp = input_ids + output_ids
                if len(inp) > Config.sequence_len - 20:
                    exced_lines.append([input, output])
                else:
                    train_datas.append({"input":input, "output":output})
            print(len(exced_lines))
        self.recheck_xls(exced_lines)
        with open(self.train_json, "w", encoding="utf-8") as jsonfile:
            json.dump(train_datas, jsonfile, ensure_ascii=False,indent=4)
    
    def recheck_xls(self, exced_lines):
        wb = Workbook()
        ws = wb.active
        ws.append(["问题", "回答"])
        
        for row in exced_lines:
            ws.append(row)
        wb.save(self.out_csv)

def get_all_data(ori_dir, target_json):
    """
    针对xls文件解析
    
    Args:
        ori_dir 原始目录包含多个xls文件
        target_json 目标json
    """
    all_data = []
    for root, dirs, files in os.walk(ori_dir):
        for file in files:
            abs_filepath = os.path.abspath(os.path.join(root, file))
            df = pd.read_excel(abs_filepath)
            for i, row in df.iterrows():
                single_data = row["json"]
                single_data = eval(single_data)
                print(single_data)
                all_data.append({"input":single_data["input"], "output":single_data["output"]})
    with open(target_json, "w", encoding="utf-8") as jsonfile:
            json.dump(all_data, jsonfile, ensure_ascii=False,indent=4)

# if __name__ == '__main__':
#     in_csv = "/home/mai-llm-train-service/dataset/data_20240614161119.csv"
#     tokenizer = AutoTokenizer.from_pretrained(
#             Config.base_model,
#             padding_side="right",
#             model_max_length=Config.sequence_len,
#             trust_remote_code=True
#             )
#     out_csv = "/home/mai-llm-train-service/dataset/recheck_v20240614_1.xlsx"
#     train_json = "/home/mai-llm-train-service/dataset/webdoctor-traindata-v20240614.json"
#     CSVTransfer(in_csv, tokenizer, out_csv, train_json).read_and_recompose()