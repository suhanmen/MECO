import json
import sys
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import pdb
import random

'''
python codecomplex_data_sampling.py 0.1 > 0.1.log
'''


step1_percentage = sys.argv[1] #step1 data percentage [0.1, 0.2, 0.3]

# Initialize data
java_data_list = []
python_data_list = []

output_java_professional = []
output_python_professional = []

java_count_professional = {'constant': 0, 'logn': 0, 'linear': 0, 'nlogn': 0, 'quadratic': 0, 'cubic': 0, 'np': 0}
python_count_professional = {'constant': 0, 'logn': 0, 'linear': 0, 'nlogn': 0, 'quadratic': 0, 'cubic': 0, 'np': 0}


# Load data
with open("./full/java_data.jsonl", "r") as f:
    for line in f:
        java_data_list.append(json.loads(line))

with open("./full/python_data.jsonl", "r") as f:
    for line in f:
        python_data_list.append(json.loads(line))


for idx, data_pl in enumerate([java_data_list, python_data_list]):
    if idx == 0:
        java_train, java_test = train_test_split(data_pl, train_size=0.9, test_size=0.1,  random_state=42)
    else:
        python_train, python_test = train_test_split(data_pl, train_size=0.9, test_size=0.1,  random_state=42)


for PL_data, PL_name in [(java_train, 'java_train'), (java_test,'java_test'), (python_train,'python_train'),(python_test,'python_test')]:
    with open(f"./codecomplex_data_sampling/{PL_name}.jsonl", "w", encoding="utf-8") as f:
        for i in PL_data:
            f.write(json.dumps(i))
            f.write("\n")


# How many samples for each class
total_sample = int(len(java_train)) * float(step1_percentage) # 0.05, 0.1, 0.15, 0.2
sampling = total_sample // 7

if len(java_test) == len(python_test):
    test_len = len(java_test)


# Create directory
dataset_paths = {
    "normal": f"./codecomplex_data_sampling/step2,3-{str(test_len)}",
    "professional": f"./codecomplex_data_sampling/step1-{int(float(step1_percentage)*100)}%-professional"
}

for path in dataset_paths.values():
    Path(path).mkdir(parents=True, exist_ok=True)


seed = int(sys.argv[2])


if step1_percentage == 0.1:
    random.seed(11)
    random.shuffle(java_train)
    random.seed(75)
    random.shuffle(python_train)
elif step1_percentage == 0.2:
    random.seed()
    random.shuffle(java_train)
    random.seed()
    random.shuffle(python_train)
elif step1_percentage == 0.3:
    random.seed(41)
    random.shuffle(java_train)
    random.seed()
    random.shuffle(python_train)

# Sampling data
for data in java_train:
    if java_count_professional[data['complexity']] < int(sampling):
        output_java_professional.append(data)
        java_count_professional[data['complexity']] += 1

for data in python_train:
    if python_count_professional[data['complexity']] < int(sampling):
        output_python_professional.append(data)
        python_count_professional[data['complexity']] += 1


print(f"test java len : {len(java_test)}")
print(f"test python len : {len(python_test)}")
print(f"output_java_professional: {len(output_java_professional)}")
print(f"output_python_professional: {len(output_python_professional)}\n")

# Save data
data_lists = {
    "normal": [java_test, python_test],
    "professional": [output_java_professional, output_python_professional]
}

for category, data_list in data_lists.items():
    for idx, data in enumerate(data_list):
        PL_name = "java" if idx == 0 else "python"
        base_path = dataset_paths[category]

        # Save JSONL
        with open(f"{base_path}/{PL_name}-multi_test_data_check.jsonl", "w", encoding="utf-8") as f:
            for i in data:
                f.write(json.dumps(i, indent=4))
                f.write("\n")

        with open(f"{base_path}/{PL_name}-multi_test_data.jsonl", "w", encoding="utf-8") as f:
            for i in data:
                f.write(json.dumps(i))
                f.write("\n")


print("***Make data done!")
