from trainNN import NeuralNetwork
import os
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path of json config file")
args = parser.parse_args()

config = {}

if args.config:
    with open(args.config) as json_file:
        config = json.load(json_file)
else:
    print("Please enter config path.")
    exit(0)

net = NeuralNetwork(config)
for i in range(10):
    batch = next(net.train_it)
    count = 0
    if(not os.path.exists('dump/train/batch{}'.format(i))):
        os.makedirs('dump/train/batch{}'.format(i))
    for j in range(64):
        if(batch[1][j][0] == 1.0):
            plt.imsave('dump/train/batch{}/{}_{}.png'.format(i,
                                                             count, "free"), batch[0][j])
        else:
            plt.imsave('dump/train/batch{}/{}_{}.png'.format(i,
                                                             count, "occupied"), batch[0][j])
        count = count + 1
for i in range(10):
    batch = next(net.val_it)
    count = 0
    if(not os.path.exists('dump/validate/batch{}'.format(i))):
        os.makedirs('dump/validate/batch{}'.format(i))
    for j in range(64):
        if(batch[1][j][0] == 1.0):
            plt.imsave('dump/validate/batch{}/{}_{}.png'.format(i,
                                                                count, "free"), batch[0][j])
        else:
            plt.imsave('dump/validate/batch{}/{}_{}.png'.format(i,
                                                                count, "occupied"), batch[0][j])
        count = count + 1
for i in range(10):
    batch = next(net.test_it)
    count = 0
    if(not os.path.exists('dump/test/batch{}'.format(i))):
        os.makedirs('dump/test/batch{}'.format(i))
    for j in range(64):
        if(batch[1][j][0] == 1.0):
            plt.imsave('dump/test/batch{}/{}_{}.png'.format(i,
                                                            count, "free"), batch[0][j])
        else:
            plt.imsave('dump/test/batch{}/{}_{}.png'.format(i,
                                                            count, "occupied"), batch[0][j])
        count = count + 1
