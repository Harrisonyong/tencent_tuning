#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2024/05/17 13:37:52
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   Config
'''

class Config:
    epochs = 30
    log_every = 20
    eval_steps = 50
    checkpoint_every = 100
    # train_steps = 50
    save_total_limit=10
    
    # micro_batch_size
    micro_batch_size = 2
    micro_eval_size = 2
    evaluation_strategy="steps"
    # gradient_accu_steps
    accu_steps = 4
    #dataclass
    val_set_size = 400
    sequence_len = 920

    # lr, scheduler,adamw,weight_decay设置
    learning_rate = 1e-4
    lr_scheduler_type="cosine"
    # scheduler warmup
    warmup_steps = 100
    weight_decay = 0.01
    adam_beta1=0.999
    adam_beta2 = 0.95

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj"]
    
    deepspeed_config = "/home/mai-llm-train-service/yql/qwen_tuning/deepspeed_config.json"
    data_path = "/home/mai-llm-train-service/dataset/webdoctor-traindata-v20240614.json"
    base_model = "/home/mai-llm-train-service/tencent/hunyuan_7b"
    output_path = "./checkpoints"
    