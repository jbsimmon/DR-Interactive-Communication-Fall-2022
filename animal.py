import json
import requests
import gensim.downloader as api
print("load wv")
wv = api.load('word2vec-google-news-300')
print("wv loaded")

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
    go = {"Name": obj + str(OBJ_ID), "Type": obj, "Location": [0, 2*OBJ_ID, 0], "Color": "default", "Size": "default"}
    for i in adj:
        if i in ["red", "blue", "green", "white"]:
            go["Color"] = i
        if i in ["big", "medium", "small"]:
            go["Size"] = i
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


def process(text):
    global OBJ_ID
    global OBJ_LIST

    OP = ["create", "delete"]
    TYPE = ["Chicken", "Cow", "Duck", "Pig", "Sheep"]

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
                l = []
                for i in OP:
                    l.append(wv.similarity(word["word"], i))
                mv = max(l)
                if mv > 0.1:
                    mi = l.index(mv)
                    op = OP[mi]
            if word["entity_group"] == "NOUN":
                l = []
                for i in TYPE:
                    l.append(wv.similarity(word["word"], i))
                mv = max(l)
                if mv > 0.1:
                    mi = l.index(mv)
                    ob = TYPE[mi]
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


if __name__ == '__main__':
    # app.run()
    print(process("give me a hen and an ox"))
    print(process("delete cow"))
