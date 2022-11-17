import json
import requests
# import gensim.downloader as api
from flask import Flask
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/vblagoje/bert-english-uncased-finetuned-pos"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

module_path = "F:/universal-sentence-encoder_4"
sim_model = hub.load(module_path)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

OBJ_ID = 1

OBJ_LIST = []

def create_obj(obj, adj):
    global OBJ_ID
    global OBJ_LIST
    go = {"Name": obj + str(OBJ_ID), "Type": obj, "Location": [0, 2*OBJ_ID, 0], "Color": "default", "Size": "default"}
    go["Location"] = adj["Location"]
    go["Color"] = adj["Color"]
    go["Size"] = adj["Size"]
    OBJ_LIST.append(go)
    OBJ_ID += 1
    return go

def delete_obj(obj, adj):
    global OBJ_LIST
    target = []
    for i in OBJ_LIST:
        if i["Type"] == obj:

            target.append(i["Name"])
    return target

def find_adj(part):
    colors = ["black", "blue", "cyan", "gray", "green", "magenta", "red", "white", "yellow"]
    sizes = ["small tiny mini", "medium standard normal", "large huge big"]
    locations = ["far", "close", "front", "back", "right", "left", "high", "low"]
    adj = {"Color": None, "Size": None, "Location": [0, 0, 0]} 
    c = [part] + colors
    embeddings = sim_model(c)
    similarity = cosine_similarity(embeddings, embeddings)[0][1:]
    mv = max(similarity)
    if mv > 0.1:
        mi = list(similarity).index(mv)
        adj["Color"] = colors[mi]

    s = [part] + sizes
    embeddings = sim_model(s)
    similarity = cosine_similarity(embeddings, embeddings)[0][1:]
    mv = max(similarity)
    if mv > 0.1:
        mi = list(similarity).index(mv)
        adj["Size"] = sizes[mi].split()[0]

    l = [part] + locations
    embeddings = sim_model(l)
    similarity = cosine_similarity(embeddings, embeddings)[0][1:]
    # distance
    if similarity[0] > similarity[1]:
        adj["Location"][1] = 5
    else:
        adj["Location"][1] = 1
    # height
    if similarity[6] > similarity[7]:
        adj["Location"][0] = 5
    else:
        adj["Location"][0] = 1
    # direction
    mv = max(similarity[2:6])
    if similarity[2] == mv:
        adj["Location"][2] = 0
    elif similarity[3] == mv:
        adj["Location"][2] = 180
    elif similarity[4] == mv:
        adj["Location"][2] = 90
    elif similarity[5] == mv:
        adj["Location"][2] = 270

    return adj






def process(text):
    global OBJ_ID
    global OBJ_LIST

    OP = ["create", "delete"]
    TYPE = ["chicken", "cow", "duck", "pig", "sheep"]

    create_list = []
    delete_list = []
    tags = query({"inputs": text})
    conj = []
    # divide text
    for i in range(len(tags)):
        if tags[i]['entity_group'] == "CCONJ":
            conj.append(i)
    parts = []
    for i in reversed(conj):
        tags, pt = tags[:i], tags[i:]
        parts.insert(0, pt)
    parts.insert(0, tags)
    pre = None
    for p in parts:
        op = None
        ob = None
        adj = []
        for word in p:
            if word["entity_group"] == "VERB":
                l = [word["word"], "create", "delete"]
                embeddings = sim_model(l)
                similarity = cosine_similarity(embeddings, embeddings)[0][1:]
                mv = max(similarity)
                if mv > 0.1:
                    mi = list(similarity).index(mv)
                    op = OP[mi]
            if word["entity_group"] == "NOUN":
                l = [word["word"], "chicken", "cow", "duck", "pig", "sheep"]
                embeddings = sim_model(l)
                similarity = cosine_similarity(embeddings, embeddings)[0][1:]
                mv = max(similarity)
                if mv > 0.1:
                    mi = list(similarity).index(mv)
                    ob = TYPE[mi]

        if ob and op == "create":
            adj = find_adj(" ".join([i["word"] for i in p]))
            create_list.append(create_obj(ob, adj))
            pre = op
        elif ob and op == "delete":
            result = delete_obj(ob, adj)
            if result:
                for i in result:
                    delete_list.append(i)
            pre = op
        elif ob and pre == "create":
            adj = find_adj(" ".join([i["word"] for i in p]))
            create_list.append(create_obj(ob, adj))
        elif ob and pre == "delete":
            result = delete_obj(ob, adj)
            if result:
                for i in result:
                    delete_list.append(i)

    rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
    return rt

app = Flask(__name__)
@app.route('/<name>')
def idx(name):

    return process(name[1:])

if __name__ == '__main__':
    app.run()
    # print(process("give me a hen and an ox"))
    # print(process("delete cow"))
