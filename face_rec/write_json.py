import json
import os

def write_json(info):
    with open('data.json','a+') as outfile:
        outfile.seek(0)
        if not outfile.read():  # if json is empty, initial
            data = {}
            data['records'] = []
            data['records'].append(info)
            json.dump(data, outfile, sort_keys=True, indent=4)
        else:
            count = 0 # checks for duplication
            with open('data.json') as json_file:
                data = json.load(json_file)
                temp = data['records']
                last_item = temp[-1]
                if last_item == info:
                    count += 1   # if 1 => record already exists
                else:
                    count += 0
                    temp.append(info)
            if count <= 0: # if record does not exist, then write into file
                update_json(data)

# updates current list of records
def update_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent = 4)