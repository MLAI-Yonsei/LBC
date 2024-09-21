import os
import pdb
from typing import List, Tuple
import numpy as np
from functools import partial
import pandas as pd
import random
import sys
sys.path.append("")

from dictionary import DatasetLike

sys.path.append('')

random_words = [
    "Good luck ", "Fighting ", "You can do it ", "Keep going ", "Don't give up ", 
    "Stay strong ", "Believe in yourself ", "Keep pushing ", "Go for it ",
]


def data2text_feature_name_Blood(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Breast_Cancer(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Creditcard(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_German(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_ILPD(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Loan(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Salary(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Steel_Plate(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if not icl:
        completion = "No" if row["y"] == 0 else "Yes"
        
        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_CMC(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''
    
    if row['y'] == 0:
        completion = 'High'
    elif row['y'] == 1:
        completion = 'Low'
    else:
        completion = 'Medium'

    if not icl:
        prompt += "What is Engagement Level? "

        prompt += random.choice(random_words)

        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_Restaurant(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if row['y'] == 0:
        completion = 'High'
    elif row['y'] == 1:
        completion = 'Low'
    else:
        completion = 'Medium'

    if not icl:
        prompt += "What is Engagement Level? "

        prompt += random.choice(random_words)

        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."

def data2text_feature_name_OGB(row, cols, train_quartiles, test_quartiles, categorical, mode='train', icl=False):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    if row['y'] == 0:
        completion = 'High'
    elif row['y'] == 1:
        completion = 'Low'
    else:
        completion = 'Medium'

    if not icl:
        prompt += "What is Engagement Level? "

        prompt += random.choice(random_words)

        return "{\"prompt\":\"%s\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    return prompt + "DESCRIPTION OF ANSWER HERE, WHICH IS USED TO ICL EXAMPLE."




def df2jsonl_feat_name(df:pd.DataFrame, filename:str, did:DatasetLike, train_quartiles:pd.DataFrame, test_quartiles:pd.DataFrame, integer:bool = False):
    fpath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)

    if did == 'Blood':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Blood, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore
    
    elif did == 'Breast_Cancer':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Breast_Cancer, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore
    
    elif did == 'Creditcard':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Creditcard, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
    
    elif did == 'German':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_German, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore

    elif did == 'ILPD':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_ILPD, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore

    elif did == 'Loan':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Loan, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore
    
    elif did == 'Salary':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Salary, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore
    
    elif did == 'Steel_Plate':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Steel_Plate, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist()) # type: ignore
    
    elif did == 'CMC':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_CMC, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())

    elif did == 'Restaurant':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Restaurant, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
    
    elif did == 'OGB':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_OGB, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
    
    else: raise NotImplementedError
    
    with open(fpath, 'w') as f:
        f.write(jsonl)
    return fpath
    
def df2jsonl_feat_name_icl(df:pd.DataFrame, filename:str, did:DatasetLike, train_quartiles:pd.DataFrame, test_quartiles:pd.DataFrame, integer:bool = False):
    fpath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    
    labels:List[bool] = (df['y'] == 1).tolist()
    def get_partitions(results:List[str])->Tuple[List[str], List[str]]:
        def p(label): return [result for i, result in enumerate(results) if labels[i]==label]
        return p(0), p(1)
    
    icl_examples:List[str]
    if did == 'Blood':
        test_prompts = df.apply(func = partial(data2text_feature_name_Blood, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Blood, mode = 'train' ,train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'Breast_Cancer':
        test_prompts = df.apply(func = partial(data2text_feature_name_Breast_Cancer, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Breast_Cancer, train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'Creditcard':
        test_prompts = df.apply(func = partial(data2text_feature_name_Creditcard, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Creditcard, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'German':
        test_prompts = df.apply(func = partial(data2text_feature_name_German, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_German, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore

    elif did == 'ILPD':
        test_prompts = df.apply(func = partial(data2text_feature_name_ILPD, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_ILPD, mode = 'train', cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore

    elif did == 'Loan':
        test_prompts = df.apply(func = partial(data2text_feature_name_Loan, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Loan, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'Salary':
        test_prompts = df.apply(func = partial(data2text_feature_name_Salary, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Salary, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'Steel_Plate':
        test_prompts = df.apply(func = partial(data2text_feature_name_Steel_Plate, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Steel_Plate, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=True), axis = 1).tolist() # type: ignore
    
    elif did == 'CMC':
        test_prompts = df.apply(func = partial(data2text_feature_name_CMC, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_CMC, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=False), axis = 1).tolist() # type: ignore
    
    elif did == 'OGB':
        test_prompts = df.apply(func = partial(data2text_feature_name_OGB, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_OGB, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=False), axis = 1).tolist() # type: ignore
    
    elif did == 'Restaurant':
        test_prompts = df.apply(func = partial(data2text_feature_name_Restaurant, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist() # type: ignore
        icl_examples = df.apply(func = partial(data2text_feature_name_Restaurant, mode = 'train', train_quartiles = train_quartiles, test_quartiles = test_quartiles, icl=False), axis = 1).tolist() # type: ignore

    else:
        raise NotImplementedError
    
    partition_neg, partition_pos = get_partitions(icl_examples)
    jsonl = []
    for test_prompt in test_prompts:
        icl_neg = random.sample(partition_neg, k=20)
        icl_pos = random.sample(partition_pos, k=10)
        
        if test_prompt in icl_neg: 
            icl_neg.pop(icl_neg.index(test_prompt))
        else: 
            icl_neg = icl_neg[:-1]
            
        if test_prompt in icl_pos: 
            icl_pos.pop(icl_pos.index(test_prompt))
        else: 
            icl_pos = icl_pos[:-1]
            
        print(len(icl_neg))
        print(len(icl_pos))
        
        # icl_neg와 icl_pos를 섞기
        icl_combined = icl_neg + icl_pos
        random.shuffle(icl_combined)
        
        icl_prompt = "{" + f"\"prompt\":\"{' '.join(icl_combined)}"
        jsonl.append(icl_prompt +  test_prompt[11:]) # Concatenate json format, deleting '}{' in  'examples...}{...test prompt' 11 is fixed (length of "{...")
    
    jsonl = '\n'.join(jsonl)
    with open(fpath, 'w') as f:
        f.write(jsonl)
    return fpath
