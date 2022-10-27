from flask import Flask
import json
import requests
import gensim.downloader as api
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch
import logging
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import time
import math

print("load wv")
wv = api.load('word2vec-google-news-300')
print("wv loaded")

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
input_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/vblagoje/bert-english-uncased-finetuned-pos"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


OBJ_ID = 1

OBJ_LIST = []


def create_obj(obj, adj):
    global OBJ_ID
    global OBJ_LIST
    go = {"Name": "object" + str(OBJ_ID), "Type": obj, "Location": "default"}
    if adj["color"]:
        go["Color"] = adj["color"]
    else:
        go["Color"] = "white"
    if adj["size"]:
        go["Size"] = adj["size"]
    else:
        go["Size"] = "medium"
    OBJ_LIST.append(go)
    OBJ_ID += 1
    return go


def delete_obj(obj, adj):
    global OBJ_LIST
    target = []
    for i in OBJ_LIST:
        if i["Type"] == obj:
            if adj["color"]:
                if adj["color"] != i["Color"]:
                    continue
            if adj["size"]:
                if adj["size"] != i["Size"]:
                    continue
            target.append(i["Name"])
    return target


def process(text):
    global OBJ_ID
    global OBJ_LIST

    op_choice = ["create", "delete"]
    ob_choice = ["cube", "sphere"]
    adj_choice = {"color": ["red", "blue", "green", "white"], "size": ["big", "medium", "small"]}
    create_list = []
    # modify_list = []
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
        adj = {"color": None, "size": None}
        for word in p:
            if word["entity_group"] == "VERB":
                l = []
                for i in op_choice:
                    l.append(wv.similarity(word["word"], i))
                mv = max(l)
                if mv > 0.1:
                    mi = l.index(mv)
                    op = op_choice[mi]
            if word["entity_group"] == "NOUN":
                l = []
                for i in ob_choice:
                    l.append(wv.similarity(word["word"], i))
                mv = max(l)
                if mv > 0.1:
                    mi = l.index(mv)
                    ob = ob_choice[mi]
            if word["entity_group"] == "ADJ":
                cat = None
                l = {}
                for i in adj_choice:
                    l[i] = wv.similarity(word["word"], i)
                mv = max(l.values())
                if mv > 0.1:
                    for k in l.keys():
                        if l[k] == mv:
                            cat = k
                            break
                if cat:
                    nl = []
                    nadj = None
                    for i in adj_choice[cat]:
                        nl.append(wv.similarity(word["word"], i))
                    nmv = max(nl)
                    if mv > 0.4:
                        mi = nl.index(nmv)
                        nadj = adj_choice[cat][mi]
                    adj[cat] = nadj

        if ob and op == "create":
            create_list.append(create_obj(ob, adj))
            pre = op
        elif ob and op == "delete":
            result = delete_obj(ob, adj)
            if result:
                for i in result:
                    delete_list.append(i)
            pre = op
        elif ob and pre == "create":
            create_list.append(create_obj(ob, adj))
        elif ob and pre == "delete":
            result = delete_obj(ob, adj)
            if result:
                for i in result:
                    delete_list.append(i)

    rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
    return rt


module_path = "universal-sentence-encoder_4"
sim_model = hub.load(module_path)

# sets of options
types = ["sphere ball", "cube box", "cylinder can", "capsule pill"]
colors = ["black", "blue", "cyan", "gray", "green", "magenta", "red", "white", "yellow"]
sizes = ["small tiny mini", "medium standard normal", "large huge big"]
locations = ["far", "close", "front", "back", "right", "left", "high", "low"]


def num_process(text):
    # require completion

    create_list = []
    # modify_list = []
    delete_list = {}
    sentences = [text] + types + colors + sizes + locations
    embeddings = sim_model(sentences)
    similarity = cosine_similarity(embeddings, embeddings)[0]

    type_upper_bound = len(types) + 1
    color_upper_bound = type_upper_bound + len(colors)
    size_upper_bound = color_upper_bound + len(sizes)

    # type calculations
    type_arr = similarity[1:type_upper_bound]
    type_max = -1
    type_idx = 0
    for i in range(len(types)):
        if type_arr[i] > type_max:
            type_max = type_arr[i]
            type_idx = i
    new_type = sentences[1 + type_idx].split(" ")[0]

    # color calculations
    color_arr = similarity[type_upper_bound:color_upper_bound]
    color_max = -1
    color_idx = 0
    for i in range(len(colors)):
        if color_arr[i] > color_max:
            color_max = color_arr[i]
            color_idx = i

    # variable size calculations
    size_arr = similarity[color_upper_bound:size_upper_bound]
    size_total_sim = sum(map(abs, size_arr))
    size_calc = 60.0
    small_calc = ((size_arr[0] * (1 - size_arr[1] / size_total_sim)) * 60)
    large_calc = ((size_arr[2] * (1 - size_arr[1] / size_total_sim)) * 60)
    if small_calc < 0:
        size_calc += abs(small_calc) ** 1.6
    else:
        size_calc -= small_calc ** 1.6
    if large_calc < 0:
        size_calc -= abs(large_calc) ** 1.6
    else:
        size_calc += large_calc ** 1.6
    size_calc /= 30
    if size_calc < 0.05:
        size_calc = 0.05

    # variable location calculations
    location_arr = similarity[size_upper_bound:]
    i = 0
    while i < len(location_arr):
        smaller = location_arr[i]
        larger = location_arr[i + 1]
        if smaller < 0:
            location_arr[i + 1] -= smaller
            location_arr[i] -= smaller
        if larger < 0:
            location_arr[i] -= larger
            location_arr[i + 1] -= larger
        i += 2

    distance = (60.0 + (location_arr[0] * 60) ** 1.6 - (location_arr[1] * 60) ** 1.6) / 6.0
    height = (3.0 + (distance * location_arr[6] * 3) - (distance * location_arr[7] * 3))

    x = distance * (0.0 + (location_arr[2] * 40) ** 1.6 - (location_arr[3] * 40) ** 1.6) / 6.0
    z = distance * (0.0 + (location_arr[4] * 40) ** 1.6 - (location_arr[5] * 40) ** 1.6) / 6.0
    direction = math.atan2(z, x) * (180 / math.pi)

    new_location = [height, distance, direction]

    print(x, " ", z, " :", direction, " d+ ", distance)
    create_list.append({"Name": "object1", "Type": new_type, "Color": sentences[type_upper_bound + color_idx],
                        "Size": size_calc, "Location": new_location})

    rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
    return rt


app = Flask(__name__)


@app.route('/<name>')
def idx(name):
    inputs = tokenizer(name[1:], return_tensors="pt")
    tf.random.set_seed(0)  # for testing purposes
    results = input_model.generate(
        **inputs,
        do_sample=True,
        top_p=0.92,
        top_k=50,
    )
    response = tokenizer.decode(results[0], skip_special_tokens=True)
    return process(response) if name[0] == '0' else num_process(response)


if __name__ == '__main__':
    app.run()
