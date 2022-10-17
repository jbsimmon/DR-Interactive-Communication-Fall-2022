from flask import Flask
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

import json

import logging
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import time

module_path = "universal-sentence-encoder_4"
sim_model = hub.load(module_path)

# sets of options
types = ["sphere ball", "cube box", "cylinder can", "capsule pill"]
colors = ["black", "blue", "cyan", "gray", "green", "magenta", "red", "white", "yellow"]
sizes = ["small tiny mini", "medium standard normal", "large huge big"]
locations = ["far", "close", "front", "back", "left", "right", "high", "low"]


def process(text):
    create_list = []
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
    small_calc = ((size_arr[0]*(1 - size_arr[1]/size_total_sim))*60)
    large_calc = ((size_arr[2]*(1 - size_arr[1]/size_total_sim))*60)
    if small_calc < 0:
        size_calc += abs(small_calc)**1.6
    else:
        size_calc -= small_calc**1.6
    if large_calc < 0:
        size_calc -= abs(large_calc)**1.6
    else:
        size_calc += large_calc**1.6
    size_calc /= 30
    if size_calc < 0.05:
        size_calc = 0.05

    # variable location calculations
    location_arr = similarity[size_upper_bound:]
    i = 0
    while i < len(location_arr):
        smaller = location_arr[i]
        larger = location_arr[i+1]
        if smaller < 0:
            location_arr[i+1] -= smaller
            location_arr[i] -= smaller
        if larger < 0:
            location_arr[i] -= larger
            location_arr[i+1] -= larger
        i += 2

    distance = (60.0 + (location_arr[0]*60)**1.6 - (location_arr[1]*60)**1.6)/6.0
    x = distance * (0.0 + (location_arr[2]*40)**1.6 - (location_arr[3]*40)**1.6)/6.0
    z = distance * (0.0 + (location_arr[4] * 40) ** 1.6 - (location_arr[5] * 40) ** 1.6) / 6.0
    y = (3.0 + (distance * location_arr[6] * 3) - (distance * location_arr[7] * 3))
    new_location = [x, y, z]

    print(color_idx)
    create_list.append({"Name": "object1", "Type": new_type, "Color": sentences[type_upper_bound + color_idx],
                        "Size": size_calc, "Location": new_location})

    rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
    return rt


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
input_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")


@app.route('/<name>')
def idx(name):
    inputs = tokenizer(name, return_tensors="pt")
    tf.random.set_seed(0)  # for testing purposes
    results = input_model.generate(
        **inputs,
        do_sample=True,
        top_p=0.92,
        top_k=50,
    )
    return process(tokenizer.decode(results[0], skip_special_tokens=True))


if __name__ == '__main__':
    app.run()