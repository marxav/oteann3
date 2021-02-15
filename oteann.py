#!/usr/bin/env python
# coding: utf-8

# ## OTEANN with Transformers
# 
# This code investigates the orthographic depth of some spelling systems.
# 
# This is a new version of OTEANN, which is now implemented with a GPT model instead of a Seq2Seq.
# 
# The code used in this pages mainly comes from https://github.com/karpathy/minGPT (under MIT licence)


# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
)


import os
import sys
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample


# make deterministic
#from mingpt.utils import set_seed
#set_seed(42)


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


# These first configuration parameters are hyperparameters that we will need to tune
CONFIG = {            
    'episodes': 3,
    'n_train': 3000,
    'n_layer': 4, 
    'n_head': 4, 
    'n_embd': 336,
    'batch_size': 256,
}

ROOT_DIR = os.getcwd()

# These other configuration parameters will not be tuned
def extend_config(config): 
    config['languages'] = ['ent', 'eno','ar', 'br', 'de', 
                           'en', 'eo', 'es', 'fi', 'fr', 
                           'fro', 'it', 'ko', 'nl', 'pt', 
                           'ru', 'sh', 'tr', 'zh']
    config['n_test'] = 1000 
    config['do_finetune'] = False 
    config['n_samples'] = config['n_train'] + config['n_test']
    config['label'] = 'oteann' + '_' + str(config['n_samples'])
    config['root_dir'] = ROOT_DIR    
    config['output_max_len'] = 25
    config['block_size'] = 63
    config['do_train'] = True  # allows skipping training between multiple re-runs
    config['tasks'] = ['write', 'read']
    config['subdatasets_dir'] = 'subdatasets'  
    config['subdataset'] = 'wikt_samples.csv' # postfix from fonetik.fr 
    config['sep'] = ','
    config['features'] =  ['Language', 'Task', 'Input', 'Output']    
    config['trial_dir'] = os.getcwd() 
    config['trial_filename'] = config['trial_dir'] + '/' + config['label'] 
    config['train_filename'] = config['trial_filename'] + '_train.csv'
    config['test_filename'] = config['trial_filename'] + '_test.csv'
    config['model_filename'] = config['trial_filename'] + '_model.pt'
    config['results_filename'] = config['trial_filename'] + '_results.csv'
    config['aggregated_subdatasets'] = config['root_dir'] + '/' + config['subdataset']
    full_text = open(config['aggregated_subdatasets'], 'r').read() 
    config['chars'] = sorted(list(set(full_text)))
    return config
    

def add_samples(config, language, task):
    
    filename = config['root_dir'] + '/' + config['subdatasets_dir'] + '/' + language + '_' + config['subdataset']
    if config['episodes'] == 1:
        print('%s: processing "%s" data from %s' %(language, task, filename))
    
    wanted_samples = config['n_samples']
    
    df = pd.read_csv(filename)    
    if df.shape[0] < wanted_samples:
        print('WARNING: ', language, 'n_samples=', df.shape[0])
        if df.shape[0] > config['n_test']:
            wanted_samples = df.shape[0]
        else:
            print('ERROR: ', language, 'not enough samples')
            return

    df = df.sample(wanted_samples)
    
    df_train = pd.DataFrame(columns=config['features'])
    df_test = pd.DataFrame(columns=config['features'])
        
    # only keep 2 columns
    df = df[['Word', 'Pronunciation']]
    
    n_max = df.shape[0]
    n_test = config['n_test']
    n_train = int(n_max - n_test)
    
    n = 0
    for index, line in df.iterrows():
        
        word = line['Word']
        try:
            l_word = len(word)
            if l_word > config['output_max_len']:
                continue
        except:
            continue            
        pron = line['Pronunciation']
        try:
            l_pron = len(pron)
            if l_pron > config['output_max_len']:
                continue
        except:
            continue
        l_word = len(word)
        l_pron = len(pron)
        
        if task == 'read':
            input = word        
            output = pron 
        elif task == 'write':
            input = pron       
            output = word 
        else:
            print('ERROR: task=',task,'should not happen')
            return
        
        sample = {'Language': language, 'Task':task, 'Input': input, 'Output': output}
        if n < n_test:
            df_test = df_test.append(sample, ignore_index = True)
        else:
            df_train = df_train.append(sample, ignore_index = True)
        n += 1
            
    # append results to our train and test datasets
    df_train.to_csv(config['train_filename'], mode='a', index=False, header=False)
    df_test.to_csv(config['test_filename'], mode='a', index=False, header=False)
    
    return
    
def generate_datasets(config):
    
    # init our two datasets
    df_train = pd.DataFrame(columns=config['features'])
    df_test = pd.DataFrame(columns=config['features'])
    
    # overwrite previous files
    df_train.to_csv(config['train_filename'], index=False, header=False)
    df_test.to_csv(config['test_filename'], index=False, header=False)
    
    # fill our two datasets
    for language in config['languages']:
        for task in config['tasks']:
            add_samples(config, language, task)


def shuffle_datasets(config):
    df_train = pd.read_csv(config['train_filename'], header=None, names=config['features'])    
    df_train = df_train.sample(frac=1)
    df_train.to_csv(config['train_filename'], index=False, header=False)
    
    df_test = pd.read_csv(config['test_filename'], header=None, names=config['features'])
    df_test = df_test.sample(frac=1)
    df_test.to_csv(config['test_filename'], index=False, header=False)
    
# minimalist check of the datasets generated
def check_datasets(config, debug=False):
    df_train = pd.read_csv(config['train_filename'], header=None, names=config['features'])
    df_test = pd.read_csv(config['test_filename'], header=None, names=config['features'])
    
    for step in ['train', 'test']:
        if step == 'train':
            df = df_train
        else:
            df = df_test
            
        if debug:
            print(step, 'shape:', df.shape)
            print(step, 'Input min len:', df.Input.str.len().min())
            print(step, 'Input max len:', df.Input.str.len().max())
            print(step, 'Output min len:', df.Output.str.len().min())
            print(step, 'Output max len:', df.Output.str.len().max())
        
        assert(df.Input.str.len().max() <= config['output_max_len'])
        assert(df.Output.str.len().max() <= config['output_max_len'])


def init_train_test_datasets(config):
    generate_datasets(config)
    shuffle_datasets(config)
    check_datasets(config)


import math
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, chars, data, block_size, debug=True):
        #chars = sorted(list(set(full_data)))
        data_size, vocab_size = len(data), len(chars)
        if debug:
            print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def predict_pron(config, train_dataset, trainer, model, word):
    try:
        x = torch.tensor([train_dataset.stoi[s] for s in word], dtype=torch.long)[None,...].to(trainer.device)
        y = sample(model, x, config['output_max_len'], temperature=1.0, sample=True, top_k=10)[0]
        completion = ''.join([train_dataset.itos[int(i)] for i in y])
    except:
        e = sys.exc_info()[0]
        print('predict_pron(): error %s for word:%s' % (e, word))
        # Typically, this can happen if a tested word contains a char
        # that did not existing during the training step
        completion = 'N/A'
    return completion


def train(config):

    training_t0 = datetime.datetime.now()  
    
    block_size = config['block_size']
    
    print("config['train_filename']:", config['train_filename'])        
    text = open(config['train_filename'], 'r').read() 
    train_dataset = CharDataset(config['chars'], text, block_size, debug=True) # one line is 63 characters

    # create model
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=config['n_layer'], 
                      n_head=config['n_head'], 
                      n_embd=config['n_embd'])

    model = GPT(mconf)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ANN parameters: %d' % pytorch_total_params)
    
    # train
    tconf = TrainerConfig(max_epochs=2, batch_size=config['batch_size'], learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, 
                          final_tokens=2*len(train_dataset)*block_size,
                          num_workers=4,
                          tqdm=False) # not config['do_finetune'])
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
    training_t1 = datetime.datetime.now()  
    training_duration = training_t1 - training_t0
    print('training_duration', training_duration)
    
    torch.save(model.state_dict(), config['model_filename'])
    
    return model


def get_model(config):
    # following two lines are copied from train()
    block_size = config['block_size']
    text = open(config['train_filename'], 'r').read() 
    train_dataset = CharDataset(config['chars'], text, config['block_size']) 
                       
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,                       
                      n_layer=config['n_layer'], 
                      n_head=config['n_head'], 
                      n_embd=config['n_embd'])
    model = GPT(mconf)
    model.load_state_dict(torch.load(config['model_filename']))
    model.eval()
    return model


def test(config, task, language, df_results, debug=False):
    
    model = get_model(config)
    
    # followwing two lines are copied from train()
    block_size = config['block_size']
    text = open(config['train_filename'], 'r').read() 
    train_dataset = CharDataset(config['chars'], text, block_size) 

    
    tconf = TrainerConfig(max_epochs=2, batch_size=config['batch_size'], learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, 
                          final_tokens=2*len(train_dataset)*block_size,
                          num_workers=4)
    trainer = Trainer(model, train_dataset, None, tconf)
    
    # test
    testing_t0 = datetime.datetime.now() 
    
    df_test = pd.read_csv(config['test_filename'], header=None, names=config['features'])    

    n = 0
    n_ok = 0
    n_ko = 0
    for index, row in df_test.iterrows():
        
        if row.Task != task or row.Language != language:
            continue
        
        n += 1
        
        # build the context 
        context = language + ',' + task + ',' + row.Input + ','
        
        # get the predicted output string
        prediction_padded = predict_pron(config, train_dataset, trainer, model, context)
        
        # identify where the interesting output is in the raw output
        if prediction_padded.startswith(context):
            
            # remove unwanted prefix
            prediction_padded = prediction_padded[len(context):]     
            
            # remove unwanted postfix (i.e. remove padding)
            eos = prediction_padded.find('\n', 1)
            #eos_p = prediction_padded.find(',', 1)
            #eos_t = row.Output.find(',', 1)
            #if eos_p < 0 or eos_t < 0:
            if eos < 0:
                n_ko += 1
            else:
                #prediction = prediction_padded[:eos_p]
                #target = row.Output[:eos_t]
                prediction = prediction_padded[:eos]
                target = row.Output
                # check if prediction is same as target
                if prediction == target:
                    n_ok += 1
                else:
                    if debug and language != 'eno':
                        print('language:%s, target:%s, prediction:%s,' % (language, target, prediction))
                    n_ko += 1            
        else:
            n_ko += 1

    pctg_ok = int(n_ok/n*100)
    pctg_ko = 100 - pctg_ok
    if config['episodes'] == 1:
        print('%s %5s: n=%d, n_ok=%d, n_ko=%d => %%n_ok=%d%%' % (language, task, n, n_ok, n_ko, pctg_ok))
    testing_t1 = datetime.datetime.now()  
    test_duration = testing_t1 - testing_t0
    
    dict_res = {'lang': language, 'task':task, 'test_accuracy': n_ok/n, 
                #'training_duration': training_duration, 
                'test_duration': test_duration}
    return dict_res


def train_and_tests(config):
    
    # open the file for being able to append the results of this test
    # otherwise create a new one
    df_results = pd.DataFrame()

    for episode in range(config['episodes']):
        print('episode:', episode)
             
        init_train_test_datasets(config)
        
        # train the ANN for all available languages in the training dataset
        # i.e. multi-language training
        train(config)
        
        # test the ANN for each languages
        for language in config['languages']:
            for task in config['tasks']:
                dict_res = test(config, task, language, df_results)
                # put the results as a new line in the CSV history file
                df_res = pd.DataFrame(data = [dict_res.values()], columns = dict_res.keys())
                df_results = pd.concat([df_results, df_res], axis=0, ignore_index=True, sort=False)
                df_results.to_csv(config['results_filename'], index=None, header=True)
                
    acc = df_results.test_accuracy.mean()
    print('accuracy:%.2f' % acc)
    
    return df_results


def tests(config):

    # open the file for being able to append the results of this test
    # otherwise create a new one
    df_results = pd.DataFrame()

    for episode in range(config['episodes']):
        print('episode:', episode)
        
        # test the ANN for each languages
        for language in config['languages']:
            for task in config['tasks']:
                dict_res = test(config, task, language, df_results)
                # put the results as a new line in the CSV history file
                df_res = pd.DataFrame(data = [dict_res.values()], columns = dict_res.keys())
                df_results = pd.concat([df_results, df_res], axis=0, ignore_index=True, sort=False)
                df_results.to_csv(config['results_filename'], index=None, header=True)
    
    acc = df_results.test_accuracy.mean()
    print('accuracy:%.2f' % acc)
    
    return df_results

import random
import ray
import ray.tune as tune

def get_6_digits():
    str_digits =  ''
    for i in range(6):
        digit = random.randint(0,9)
        str_digits += str(digit)
    return(str_digits)
        
def ray_train_and_tests(config):
        
        # update config with two additional parameters related to phonemes
        config = extend_config(config) 
        config['ray_tune'] = True
        
        ray_instance = 'ray_test_'+ get_6_digits()        
        #config['trial_dir'] = os.getcwd()
        #+ '/' + config['root_label'] + '_' + ray_instance
        print(config)
        
        df_results = train_and_tests(config)
        acc = df_results.test_accuracy.mean()
        tune.report(accuracy=acc)
        
        
def finetune_hyperparameters():

    FINETUNING_CONFIG = {            
        'episodes': tune.grid_search([3]),
        'n_train': tune.grid_search([10000]),
        'n_layer': tune.grid_search([4]),
        'n_head': tune.grid_search([4]), 
        'n_embd': tune.grid_search([336]),
        'batch_size': tune.grid_search([256, 512]),        
    }
    
    test_name = 'ray_test_' + get_6_digits()
    print('test_name:%s' % test_name)

    # https://docs.ray.io/en/latest/tune/api_docs/execution.html?highlight=tune%20run
    analysis = tune.run(
        ray_train_and_tests, 
        config=FINETUNING_CONFIG, 
        resources_per_trial={'gpu': 4},
        name=test_name,
        local_dir= os.getcwd() + '/' + 'ray_results',
        metric="accuracy", 
        mode="max"
        )

    print("Best config: ", analysis.get_best_config())

    # Get a dataframe for analyzing trial results.
    df_ray = analysis.dataframe()

    return df_ray


def main():
    config = extend_config(CONFIG)
    if config['do_finetune']:
        ray.init()
        df_ray = finetune_hyperparameters()
    else:
        if config['do_train']:
            df_results = train_and_tests(config)
        else:
            df_results = tests(config)

main()
    
