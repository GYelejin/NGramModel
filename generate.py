import argparse
import re
import pickle
import os
import sys
import numpy as np
import core

def ArgumentParser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default="",
                        help='путь к файлу, из которого загружается модель.')
    parser.add_argument('--prefix',  type=str, default="",
                        help='необязательный аргумент. Начало предложения (одно или несколько слов). Если не указано, выбираем начальное слово случайно из всех слов.')
    parser.add_argument('--length',  type=int, default=8,
                        help='длина генерируемой последовательности.')
    return parser

def generate(model, prefix, length):
    model = core.model(model)
    model.load()
    print(model.predict(prefix, length))
if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    generate(args.model, args.prefix, args.length)
    