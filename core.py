import itertools
import collections
import re
import pickle
import os
import numpy as np
from sys import stdin

def anylyse_texts(iterable_texts, n = 2):
    # iterable_texts тексты в виде двойных массивов
    # n              выбор n-граммости модели
    pairs = []
    for iterable in iterable_texts:
        #create a list of n iters in group
        iters = itertools.tee(iterable, n)
        for i in range(1 , len(iters)):
            next(itertools.islice(iters[i], i-1, len(iterable)), None)
        #crete pairs from iterable groups
        for i in zip(*iters):
            pairs.append((hash(i[:-1]), i[-1]))
    

    #create a dictionary for each pair in group
    data = dict()
    for pair in pairs:
        key, value = pair
        last_value = data.get(hash(key))
        if last_value != None:
            last_value.append(value)
            data.update({hash(key): last_value})
        else:
            data.update({hash(key): [value]})

    #count objects and create probability
    final_dataset = dict()
    for key, value in data.items():
        conter = collections.Counter(value)
        probability = dict()
        for ikey, value in conter.items():
            probability.update({ikey : value / conter.total()})
        final_dataset.update({key : probability})

    return final_dataset


class TextParser():
    def __init__(self, path):
        self.path = path
        self.texts = []
    
    def russian_language_filter(self, text):
        russian_language_pattern = re.compile("[а-яА-Я]+")
        return re.findall(russian_language_pattern, text)

    def read_text_from_stdin(self):
        print("Input:\n")
        self.texts.append(self.russian_language_filter(stdin.readline().rstrip().lower()))

    def read_text_file(self, filename): 
        with open(f"{self.path}/{filename}", 'r') as file:
            return self.russian_language_filter(file.read().lower())

    def read_text_from_directory(self): 
        os.chdir(self.path)
        for filename in os.listdir():
            if filename.endswith(".txt"):
                self.texts.append(self.read_text_file(filename))
    
    def texts_prob(self, n):
        if self.path != None:
            self.read_text_from_directory()
        else:
            self.read_text_from_stdin()
        return anylyse_texts(self.texts, n) 


class model():
    def __init__(self, texts_path=None, path=None):
        self.texts_path = texts_path
        self.dataset = None
        self.model_dir = path if path != None else "mymodel.pkl"
        self.n = 2 # n-граммность модели (>4 плохо на практике так не делают)

    def fit(self):
        self.dataset = TextParser(self.texts_path).texts_prob(self.n)
        self.save()

    def predict(self, prefix='', length=10):
        result = prefix.split()
        true_prefix = prefix.split()[self.n:]
        
        for _ in range(length):
            hash_key = hash(tuple(true_prefix))
            variants = self.dataset.get(hash_key) if self.dataset.get(hash_key) != None else self.dataset.get(np.random.choice(list(self.dataset.keys()), 1)[0])
            model_pred = np.random.choice(list(variants.keys()), 1, p=list(variants.values()))[0]
            result.append(model_pred)
            true_prefix = prefix[-self.n:]

        return " ".join(result)

    def save(self):
        with open(self.model_dir, 'wb') as file:  # Выгружаю модель
            pickle.dump(self.dataset, file)
        print(f"Model saved at { self.model_dir}")

    def load(self):
        print(self)
        with open(self.model_dir, 'rb') as file:  # Загружаем модель
            self.dataset = pickle.load(file)
        print(f"Model loaded successfully from {self.model_dir}")
