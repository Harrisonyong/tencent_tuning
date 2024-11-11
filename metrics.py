#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2024/05/21 17:34:20
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   metrics
'''

def mean(item:list):
    """
    计算列表元素平均值
    :param desc: item 列表对象
    :type : list
    :return: 均值
    :rtype: int
    """
    return sum(item) / len(item) if len(item) > 0 else 0

def accuracy(pred):
    """
    计算生成的序列和真实标签的精确度，tokenbytoken的计算是否完全匹配
    
    :param desc: pred trainer_utils.EvalPrediction
    :return: dict accuracy
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    total = 0
    correct = 0
    for pred_y, true_y in zip(preds, labels):
        pred_y = pred_y[:-1]
        true_y = true_y[1:]
        
        for p, t in zip(pred_y, true_y):
            if t != -100:
                total += 1
                if p == t:
                    correct += 1
    return {'accuracy':correct /total if total >0 else 0}