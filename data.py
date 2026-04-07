import json;
import re;

with open("floors.json","r") as f:
    data = json.load(f);
for hole in data:
    for (i,floor) in enumerate(data[hole]):
        data[hole][i] = (re.sub("]\(","",re.sub("![\[【]([\]】][\(（](.+?)[\)）])?","",re.sub("##.+\n","",re.sub("https?://[^\s]+","",floor)))));
with open("data.json","w") as f:
    json.dump(data,f)