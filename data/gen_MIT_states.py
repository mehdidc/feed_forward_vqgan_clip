import random
from itertools import product

def sentence(obj, adj):
    return f"The {obj} in this picture is {adj}. {obj.capitalize()} is {adj}."

lines = open("MIT_states_raw.txt").readlines()
adjs = sorted(list(set([l.split(" ")[0].strip() for l in lines if len(l.split(" ")) == 2])))
objs = sorted(list(set([l.split(" ")[1].strip() for l in lines if len(l.split(" ")) == 2])))
texts = []
for adj, obj in product(adjs, objs):
    text = sentence(obj, adj)
    texts.append(text)
random.seed(42)
random.shuffle(texts)
nb_train = int(len(texts) * 0.9)
train = texts[:nb_train]
test = texts[nb_train:]
with open("MIT_states_train.txt", "w") as fd:
    fd.write("\n".join(train))
with open("MIT_states_test.txt", "w") as fd:
    fd.write("\n".join(test))
