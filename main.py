# main.py
import Networks.model as model
import Networks.ssl_model as ssl

import argparse
import yaml
import os
from os.path import dirname, abspath

rootDirectory = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory, "Dataset")
imgDirectory = os.path.join(datasetDirectory, "images")
maskDirectory = os.path.join(datasetDirectory, "annotations")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')
parser.add_argument('-ssl', action='store_true')


def main(args):
    # --- 0) Read yaml config (absolute path)
    yaml_path = os.path.join(rootDirectory, "Todo_List", f"{args.exp}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        param = yaml.safe_load(f)

    resultsPath = os.path.join(rootDirectory, "Results", args.exp)

    # --- 1) Instantiate network
    if args.ssl:
        myNetwork = ssl.Network_Class(param, imgDirectory, maskDirectory, resultsPath)
    else:
        myNetwork = model.Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    # --- 2) Train
    print("Start to train the network")
    myNetwork.train()
    print("The network is trained")

    # --- 3) Evaluation
    myNetwork.loadWeights()
    myNetwork.evaluate()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
