import random
from itertools import product

def sentence(obj, adj):
    return f"The {obj} in this picture is made of {adj}."

lines = open("MIT_states_raw.txt").readlines()
objs = list(set([l.split(" ")[1].strip() for l in lines if len(l.split(" ")) == 2]))
random.shuffle(objs)
objs = objs[0:9]
texts = []
for adj, obj in product(objs, objs):
    text = sentence(obj, adj)
    texts.append(text)
with open("test.txt", "w") as fd:
    fd.write("\n".join(texts))
