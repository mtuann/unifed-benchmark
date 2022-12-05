import json

config = json.load(open('config.json', 'r'))

f = open("data_config","w")
last_word = list(config["dataset"].split('_'))[-1]

if last_word == "vertical":
    if config["algorithm"] != "Hetero-LR" or config["model"] != "logistic_regression":
        print("Our vertical experiment only supports Hetero-LR using logistic_regression")
        assert 1==0
    f.write("framework=vertical")
else:
    f.write("framework=horizontal")