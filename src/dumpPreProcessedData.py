from trainNN import NeuralNetwork
import os
import matplotlib.pyplot as plt

net = NeuralNetwork(config)
for i in range(10):
    batch = next(net.train_it)
    count = 0
    os.makedirs('dump/train/batch{}'.format(i))
    for j in range(64):
        plt.imsave(
            'dump/train/batch{}/{}.png'.format(i, count), batch[0][j])
        count = count + 1
for i in range(10):
    batch = next(net.val_it)
    count = 0
    os.makedirs('dump/validate/batch{}'.format(i))
    for j in range(64):
        plt.imsave(
            'dump/validate/batch{}/{}.png'.format(i, count), batch[0][j])
        count = count + 1
for i in range(10):
    batch = next(net.test_it)
    count = 0
    os.makedirs('dump/test/batch{}'.format(i))
    for j in range(64):
        plt.imsave(
            'dump/test/batch{}/{}.png'.format(i, count), batch[0][j])
        count = count + 1
