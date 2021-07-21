import random
from itertools import product

def sentence(obj, adj):
    return f"The {obj} in this picture is made of {adj}."

lines = open("MIT_states_raw.txt").readlines()
objs = list(set([l.split(" ")[1].strip() for l in lines if len(l.split(" ")) == 2]))
adjs = objs
texts = []
for adj, obj in product(objs, adjs):
    if adj != obj:
        text = sentence(obj, adj)
        texts.append(text)
random.seed(42)
random.shuffle(texts)
nb_train = int(len(texts) * 0.9)
train = texts[:nb_train]
test = texts[nb_train:]
with open("MIT_objects_train.txt", "w") as fd:
    fd.write("\n".join(train))
with open("MIT_objects_test.txt", "w") as fd:
    fd.write("\n".join(test))
for obj in objs:
    text = '\n'.join([sentence(obj, adj) for adj in adjs if adj != obj])
    with open(obj+'.txt', 'w') as fd:
        fd.write(text)
