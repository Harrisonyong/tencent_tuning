#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   generate.py
@Time    :   2024/05/23 16:09:20
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   generate
'''
import os
import sys

project_path = os.path.abspath(".")
sys.path.insert(0, project_path)

from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


class Generator:
    def __init__(self,lora_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            lora_dir,
            torch_dtype = torch.float16,
            device_map = "auto",
            trust_remote_code=True,
        )
    
    def make_context(
        self,
        query: str,
        max_window_size: int,
        history: List[Tuple[str, str]] = None,
        system: str = "",
    ):
        nl_tokens = self.tokenizer.encode("\n")
        system_text = f"{system}<unused5>\n"
        system_tokens = self.tokenizer.encode(system_text)

        raw_text = ""
        context_tokens = []
        for turn_query, turn_response in reversed(history):
            query_text = turn_query
            query_tokens = self.tokenizer.encode(query_text) + [0]
            response_text = turn_response
            response_tokens = self.tokenizer.encode(query_text) + [20]
            next_context_tokens = query_tokens + response_tokens + nl_tokens
            prev_chat = f"{query_text}<|separator|>{response_text}<unused13>\n"
            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        context_tokens = system_tokens + context_tokens
        raw_text = f"{system_text}" + raw_text
        context_tokens += self.tokenizer.encode(query) + [0]
        raw_text += query
        return raw_text, context_tokens
    
    def evaluate(self, 
                 instruction="你是一个由微医开发的有用的人工智能助手。", 
                 input=None, 
                 temperature = 0.3,
                 top_p = 0.75,
                 top_k =20,
                 max_new_tokens = 512,
                 ):
        _, context_tokens = self.make_context(
            query=input,
            max_window_size=max_new_tokens, 
            history=[],
            system=instruction
        )
        input_ids = torch.tensor([context_tokens]).to(self.model.device)
    
        with torch.no_grad():
            generated_ids = self.model.generate(
                do_sample =True,
                input_ids = input_ids,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                top_k = top_k
            )
            generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)

if __name__ == '__main__':
    lora_dir = "/home/mai-llm-train-service/yql/tencent_tuning/checkpoints/endpoint"
    genrator = Generator(lora_dir=lora_dir)
    input = "你是？"
    res_lora = genrator.evaluate(input=input)
