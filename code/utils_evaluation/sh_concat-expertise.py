import os
import json
import sys
import pdb

'''
python sh_concat-expertise.py sample_pro-1050
'''

data_option = sys.argv[1]
generation_type = sys.argv[2]
option_task = sys.argv[3]
models = sys.argv[4]
TAG = sys.argv[5]
expertise = {}
expertise_part = {}

for language in ['java', 'python']:
    for i in models.split(','):    
        with open(f'../code/output_scoring/{generation_type}/{TAG}/{option_task}/{data_option}/{i}/{i}-{language}-expertise.json', "r", encoding="utf-8") as v:
            expertise[i] = json.load(v)
            expertise_part[i] = {}  # Initialize the time_complex assigned to each model
            expertise_part[i]['time-complex'] = []
            expertise_part[i]['score'] = []
    
    # Find the best model for each time_complex
    for time_complex in ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']:
        top_score = 0
        best_models = ''
        equal_models = []
        # Compare the performance of each model
        for i in ["deepseek", "llama", "qwen", "ministral"]:
            score = expertise[i][time_complex]
            if score > top_score:
                top_score = score
                best_models = i # Update the new best model

            elif top_score == score:
                equal_models.append(i)  # If it's a tie, add it
                

        
        # Add the time_complex to the best model
        if len(equal_models) == 0:
            expertise_part[best_models]['time-complex'].append(time_complex)
            expertise_part[best_models]['score'].append(round(top_score, 4))
        else:
            expertise_part[best_models]['time-complex'].append(time_complex)
            expertise_part[best_models]['score'].append(round(top_score, 4))
            for best_m in equal_models:
                #pdb.set_trace()
                expertise_part[best_m]['time-complex'].append(time_complex)
                expertise_part[best_m]['score'].append(round(score, 4))


    # print result (check the time_complex assigned to each model)
    for model, time_complexes in expertise_part.items():
        print(f"{language} - Model: {model}, "
              f"Best Complexity: {time_complexes}," )
    print('\n')

    with open(f'../code/output_scoring/{generation_type}/{TAG}/{option_task}/{data_option}/{language}-expertise.json', "w", encoding="utf-8") as v:
        json.dump(expertise_part, v, indent=4)
    

