from flask import Flask
import json
import requests
import gensim.downloader as api
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print("load wv")
wv = api.load('word2vec-google-news-300')
print("wv loaded")

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")


API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/vblagoje/bert-english-uncased-finetuned-pos"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

OBJ_ID = 1

OBJ_LIST = []

# def process(text):
#     global OBJ_ID
#     global OBJ_LIST
#
#     create_list = []
#     # modify_list = []
#     delete_list = []
#     sentences = [text, "sphere", "cube", "red", "blue", "green", "white", "small", "medium", "large", "create an object", "delete an object"]
#     embeddings = sim_model(sentences)
#     similarity = cosine_similarity(embeddings, embeddings)[0]
#     print(similarity)
#
#     op_arr = similarity[10:]
#     op_max = 0.1
#     op_idx = -1
#     for i in range(2):
#         if op_arr[i] > op_max:
#             op_max = op_arr[i]
#             op_idx = i
#     if op_idx == -1:
#         # not create or delete
#         rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
#         return rt
#     elif op_idx == 1:
#         # delete stuff
#         for o in OBJ_LIST:
#             delete_list.append(o["Name"])
#         rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
#         OBJ_LIST = []
#         return rt
#
#     color_arr = similarity[3:7]
#     color_max = -1
#     color_idx = 0
#     for i in range(4):
#         if color_arr[i] > color_max:
#             color_max = color_arr[i]
#             color_idx = i
#
#     size_arr = similarity[7:10]
#     size_max = -1
#     size_idx = 0
#     for i in range(3):
#         if size_arr[i] > size_max:
#             size_max = size_arr[i]
#             size_idx = i
#
#     type_arr = similarity[1:3]
#     type_max = -1
#     type_idx = 0
#     for i in range(2):
#         if type_arr[i] > type_max:
#             type_max = type_arr[i]
#             type_idx = i
#
#     go = {"Name": "object"+str(OBJ_ID), "Type": sentences[1+type_idx], "Color": sentences[3+color_idx], "Size": sentences[7+size_idx], "Location": "default"}
#     create_list.append(go)
#     OBJ_LIST.append(go)
#     OBJ_ID += 1
#
#     rt = json.dumps({"Text": text, "Object":{"Create": create_list, "Delete": delete_list}})
#     return rt

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




app = Flask(__name__)
@app.route('/<name>')
def idx(name):
    inputs = tokenizer(name, return_tensors="pt")
    results = model.generate(**inputs)
    response = tokenizer.decode(results[0], skip_special_tokens=True)
    return process(response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()
    # print(process("I want you to give me a yellow ball and a blue cube"))
