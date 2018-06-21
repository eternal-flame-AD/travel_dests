import tensorflow as tf
import json
import random
import copy
import os

BATCH_SIZE = 400
TEST_SIZE = 500

def process_json_data(data):
    res = []
    wght = []
    for people in data:
        res.append([x['city'] for x in people['footpath']])
        if len(res[-1])<2:
            res.pop()
        else:
            wght.append(len(res[-1]))
    return (res, wght)

CITY_LIST = []
CITY_PROB = []

def init_city_list(data):
    global CITY_LIST
    res = set(CITY_LIST)
    for people in data:
        for city in people:
            res.add(city)
    CITY_LIST = list(res)

DATA = process_json_data(json.load(open("dests.json")))
init_city_list(DATA[0])
print("Valid data count:", len(DATA[0]))
CITY_COUNT = len(CITY_LIST)
CITY_ENUM = {CITY_LIST[i]:i for i in range(CITY_COUNT)}
CITY_DATA_LIST = []

def init_city_prob(data):
    global CITY_PROB
    CITY_PROB = [0 for _ in CITY_LIST]
    s=0
    for people in data:
        for city in people:
            CITY_PROB[CITY_ENUM[city]]+=1
            s+=1
    for i in range(len(CITY_PROB)):
        CITY_PROB[i] = CITY_PROB[i]/s

init_city_prob(DATA[0])


if not os.path.exists("./city_list.txt"):
    with open("./city_list.txt", mode="w") as f:
        for city in CITY_LIST:
            f.write(city)
            f.write("\n")

TEST_DATA = (DATA[0][:TEST_SIZE],DATA[1][:TEST_SIZE])
DATA = (DATA[0][TEST_SIZE:],DATA[1][TEST_SIZE:])

def create_layer_from_city_list(li, value=1 ,default=0):
    res=[default for _ in range(CITY_COUNT)]
    for city in li:
        res[CITY_ENUM[city]] = value
    return res

def city_id_to_name(city_id):
    items = CITY_ENUM.items()
    for item in items:
        if item[1]==city_id:
            return item[0]

def transform_enum_list_to_cities(li):
    res=[]
    for city_id in li:
        res.append(city_id_to_name(city_id))
    return res

def feed_data():
    batch_x = []
    batch_y = []
    for _ in range(BATCH_SIZE):
        all_dests = copy.copy(random.choices(DATA[0], weights=DATA[1], k=1)[0])
        x = [0 for _ in range(CITY_COUNT)]
        for dest in all_dests:
            x[CITY_ENUM[dest]] = 1
        y = [CITY_PROB[i] for i in range(CITY_COUNT)]
        for city in all_dests:
            y[CITY_ENUM[city]] = 0
        del_count = random.randint(1, len(all_dests)-1)
        cities_to_del = random.choices(all_dests, k=del_count)
        for city in cities_to_del:
            x[CITY_ENUM[city]] = 0
            y[CITY_ENUM[city]] = 1
        batch_x.append(x)
        batch_y.append(y)
    return batch_x, batch_y

def feed_test_data():
    batch_x = []
    batch_y = []
    for _ in range(BATCH_SIZE):
        all_dests = copy.copy(random.choices(TEST_DATA[0], weights=TEST_DATA[1], k=1)[0])
        x = [0 for _ in range(CITY_COUNT)]
        for dest in all_dests:
            x[CITY_ENUM[dest]] = 1
        y = [CITY_PROB[i] for i in range(CITY_COUNT)]
        for city in all_dests:
            y[CITY_ENUM[city]] = 0
        del_count = random.randint(1, len(all_dests)-1)
        cities_to_del = random.choices(all_dests, k=del_count)
        for city in cities_to_del:
            x[CITY_ENUM[city]] = 0
            y[CITY_ENUM[city]] = 1
        batch_x.append(x)
        batch_y.append(y)
    return batch_x, batch_y
