import json
import sys
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Initialize data
java_data_list = []
python_data_list = []

few_shot_java_train = {}
few_shot_python_train = {}

# Load data
with open("./full/java_data.jsonl", "r") as f:
    for line in f:
        java_data_list.append(json.loads(line))

with open("./full/python_data.jsonl", "r") as f:
    for line in f:
        python_data_list.append(json.loads(line))

# Split data
java_train, java_test = train_test_split(java_data_list, train_size=0.9, test_size=0.1, random_state=42)
python_train, python_test = train_test_split(python_data_list, train_size=0.9, test_size=0.1, random_state=42)

random.seed(45)
random.shuffle(java_train)
random.shuffle(python_train)

# Select the shortest code for each complexity
for data in java_train:
    complexity = data['complexity']
    code_length = len(data['src'])
    if complexity not in few_shot_java_train or code_length < len(few_shot_java_train[complexity]['src']) and code_length > 50:
        few_shot_java_train[complexity] = data

for data in python_train:
    complexity = data['complexity']
    code_length = len(data['src'])
    if complexity not in few_shot_python_train or code_length < len(few_shot_python_train[complexity]['src']) and code_length > 400:
        few_shot_python_train[complexity] = data

# Save JSONL
with open(f"./7shot/python-7shot.jsonl", "w", encoding="utf-8") as f:
    for data in few_shot_python_train.values():
        f.write(json.dumps(data))
        f.write("\n")

with open(f"./7shot/java-7shot.jsonl", "w", encoding="utf-8") as f:
    for data in few_shot_java_train.values():
        f.write(json.dumps(data))
        f.write("\n")

print("***Make data done!")

