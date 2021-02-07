import json
import os
from trainNN import NeuralNetwork

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path of json config file")
    parser.add_argument("--data", help="Path of Video file to label")
    parser.add_argument("--model", help="Name of model in keras_models")
    args = parser.parse_args()

    model_dir = os.path.join(os.getcwd(), '../keras_models')
    logs_dir = os.path.join(os.getcwd(), '../logs')
    results_dir = os.path.join(os.getcwd(), '../results/')

    config = {}

    # Default number of epochs
    config["epochs"] = 2

    # Default batch size
    config["batch_size"] = 100

    # Default data path
    config["data"] = '../data'

    # Default model name
    config["model"] = 'inceptionv4.h5'

    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)

    # Override data path
    if args.data:
        config["data"] = args.data

    # Override model name
    if args.model:
        config["model"] = args.model

    net = NeuralNetwork(config)
    net.compileModel()
    net.testModel()
