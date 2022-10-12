from flask import Flask
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

module_path = "F:/universal-sentence-encoder_4"
sim_model = hub.load(module_path)

OBJ_ID = 1

OBJ_LIST = []

def process(text):
    global OBJ_ID
    global OBJ_LIST

    create_list = []
    # modify_list = []
    delete_list = []
    sentences = [text, "sphere", "cube", "red", "blue", "green", "white", "small", "medium", "large", "create an object", "delete an object"]
    embeddings = sim_model(sentences)
    similarity = cosine_similarity(embeddings, embeddings)[0]
    print(similarity)

    op_arr = similarity[10:]
    op_max = 0.1
    op_idx = -1
    for i in range(2):
        if op_arr[i] > op_max:
            op_max = op_arr[i]
            op_idx = i
    if op_idx == -1:
        # not create or delete
        rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
        return rt
    elif op_idx == 1:
        # delete stuff
        for o in OBJ_LIST:
            delete_list.append(o["Name"])
        rt = json.dumps({"Text": text, "Object": {"Create": create_list, "Delete": delete_list}})
        OBJ_LIST = []
        return rt

    color_arr = similarity[3:7]
    color_max = -1
    color_idx = 0
    for i in range(4):
        if color_arr[i] > color_max:
            color_max = color_arr[i]
            color_idx = i

    size_arr = similarity[7:10]
    size_max = -1
    size_idx = 0
    for i in range(3):
        if size_arr[i] > size_max:
            size_max = size_arr[i]
            size_idx = i

    type_arr = similarity[1:3]
    type_max = -1
    type_idx = 0
    for i in range(2):
        if type_arr[i] > type_max:
            type_max = type_arr[i]
            type_idx = i

    go = {"Name": "object"+str(OBJ_ID), "Type": sentences[1+type_idx], "Color": sentences[3+color_idx], "Size": sentences[7+size_idx], "Location": "default"}
    create_list.append(go)
    OBJ_LIST.append(go)
    OBJ_ID += 1

    rt = json.dumps({"Text": text, "Object":{"Create": create_list, "Delete": delete_list}})
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
    # print(process("sure, I will give you an item, what would you like?"))
