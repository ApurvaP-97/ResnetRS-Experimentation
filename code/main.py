### YOUR CODE HERE

import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs
from Configure import preprocess_config
from Network import ResnetRS

'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()'''

if __name__ == '__main__':

    model = MyModel()
    #Since download is set to True in DataLoader.py, the CIFAR-10 data will be downloaded to the data_dir
    train_data, test_data=load_data('data_dir',preprocess_config)
    model.train(train_data,test_data,200,0.1,0.00004,0.9) #Comment this during Test and Predict mode

