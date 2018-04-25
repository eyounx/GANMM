import sys
import os
import tflib.chara
import tflib.segment
import tflib.sae_mnist
import tflib.mnist
from GANMM import GANMM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data",type=str,help="mnist, sae_mnist, chara, seg")
parser.add_argument("--scale",type=float,default=1,help="data scale, (0,1]")
args = parser.parse_args()

def inf_train_gen(train_gen):
    while True:
        for images,targets in train_gen():
            yield images,targets

if __name__=="__main__":


    if args.data=="chara":
        from nets.nets_for_UCI import *
        ganmm = GANMM(6,10,Generator,Discriminator,MNN)

        data_gen = inf_train_gen(tflib.chara.load(50,50,scale=args.scale))
        split_gen= tflib.chara.splitGenerator(50,scale=args.scale)
        full_data = tflib.chara.getTrainData()

        ganmm.train(data_gen,split_gen,full_data,log_path="Result")

    elif args.data=="seg":
        from nets.nets_for_UCI import *
        ganmm = GANMM(19,7,Generator,Discriminator,MNN)

        data_gen = inf_train_gen(tflib.segment.load(50,50,scale=args.scale))
        split_gen= tflib.segment.splitGenerator(50,scale=args.scale)
        full_data = tflib.segment.getTrainData()

        ganmm.train(data_gen,split_gen,full_data,log_path="Result")

    elif args.data=="sae_mnist":
        from nets.nets_for_sae_mnist import *
        ganmm = GANMM(10,10,Generator,Discriminator,MNN)

        data_gen = inf_train_gen(tflib.sae_mnist.load(50,50,scale=args.scale))
        split_gen= tflib.sae_mnist.splitGenerator(50,scale=args.scale)
        full_data = tflib.sae_mnist.getTrainData()

        ganmm.train(data_gen,split_gen,full_data,log_path="Result")

    elif args.data=="mnist":
        from nets.nets_for_mnist import *
        ganmm = GANMM(784,10,Generator,Discriminator,MNN)

        data_gen = inf_train_gen(tflib.mnist.load(50,50,scale=args.scale))
        split_gen= tflib.mnist.splitGenerator(50,scale=args.scale)
        full_data = tflib.mnist.getTrainData()

        ganmm.train(data_gen,split_gen,full_data,log_path="Result")
    else:
        print("Invalid argument")
