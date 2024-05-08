import json
import traceback
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
import numpy as np
import os
import time
import re
import argparse
from utils import convert_json, convert_json_to_dict_golden

def gpt4_evaluator(dimension, character, predict_data, golden_data):
    prompt = eval_prompt.format(
        character = character,
        dimension = dimension,
        golden = golden_data,
        summary = predict_data
    )
    response = None
    num_attemps = 0
    while response is None and num_attemps < 3:
        try:
            response = evaluator.invoke(
                prompt,
            ).content
        except Exception as e:
            print(traceback.format_exc())
            num_attemps += 1
            print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
            time.sleep(5)

    try:
        result = eval(response)
        return result
    except:
        start_match = re.search(r'[\{]', response)
        end_match = re.search(r'[\}]', response[::-1])
        result = response
        if start_match and end_match:
            end_index = len(response) - end_match.start()
            result = response[start_match.start():end_index]
        elif start_match and not end_match:
            if response[start_match.start()] == '[':
                result = response[start_match.start():] + ']'
            else:
                result = response[start_match.start():] + '}'
            
        try:
            result = json.loads(result)
            return result
        except:
            print("Failed to evaluate:", response)
            return response
        

def evaluation(predict_data, golden_data):
    if os.path.exists(new_file_path):
        with open(new_file_path, "r", encoding='utf-8') as f:
            gpt4_score = json.load(f)
    else:
        gpt4_score = {}
    traits_scores = []
    relationships_scores = []
    events_scores = []
    personality_scores = []
    titles = list(predict_data.keys())
    print(len(titles))
    for title in tqdm(titles):
        if title in gpt4_score:
            # print(f"skipping {title}\n")
            traits_scores.append(gpt4_score[title]['traits'])
            relationships_scores.append(gpt4_score[title]['relationships'])
            events_scores.append(gpt4_score[title]['events'])
            personality_scores.append(gpt4_score[title]['personality'])
            continue
        character = golden_data[title]['character']
        predict_trait = predict_data[title]['traits']
        golden_trait = golden_data[title]['traits']
        traits_score = gpt4_evaluator('traits', character, predict_trait, golden_trait)
        traits_scores.append(traits_score)
        
        predict_relationships = predict_data[title]['relationships']
        golden_relationships = golden_data[title]['relationships']
        relationships_score = gpt4_evaluator('relationships', character, predict_relationships, golden_relationships)
        relationships_scores.append(relationships_score)
        
        predict_events = predict_data[title]['events']
        golden_events = golden_data[title]['events']
        events_score = gpt4_evaluator('events', character, predict_events, golden_events)
        events_scores.append(events_score)
        
        predict_personality = predict_data[title]['personality']
        golden_personality = golden_data[title]['personality']
        personality_score = gpt4_evaluator('personality', character, predict_personality, golden_personality)
        personality_scores.append(personality_score)
        
        gpt4_score[title] = {
        'traits': traits_score,
        'relationships': relationships_score,
        'events': events_score,
        'personality': personality_score
        }
        
        with open(new_file_path,'w',encoding='utf-8') as f:
            def set_default(obj):
                if isinstance(obj, set):
                    return list(obj)
                raise TypeError
            json.dump(gpt4_score,f,indent=4,default=set_default)
    
    return traits_scores, relationships_scores, events_scores, personality_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", type=str, default="../exp/incremental/gpt-4-0125/result.json", help="path to the json file to save the data")
    parser.add_argument("--golden_path", type=str, default = "../data/truth_persona_all_dimension.json", help="path to the json file storing the golden reference profile")

    parser.add_argument("--type", type=str, default="incremental", help="summarization method(incremental/hierarchical/once)")
    args = parser.parse_args()
    
    pid = os.getpid()
    print("pid:", pid)
    predict_path = args.predict_path
    golden_path = args.golden_path

    dir_path = os.path.dirname(predict_path)

    new_file_path = os.path.join(dir_path, "eval_score.json")

    with open(predict_path, "r", encoding='utf-8') as f:
        predict_data = json.load(f)
    with open(golden_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        
    eval_prompt = open("prompts/consistency_score/prompt.txt", "r").read()
    
    predict_data = convert_json(predict_data, type=args.type)
    golden_data = convert_json_to_dict_golden(golden_data)
    
    evaluator = ChatOpenAI(model='gpt-4', temperature=0)

    start_time = time.time()

    with get_openai_callback() as cb:
        result = evaluation(predict_data, golden_data)
        # print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    print("predict_path", predict_path)
    print({
            'traits': np.mean([item['score'] for item in result[0]]),
            'relationships': np.mean([item['score'] for item in result[1]]),
            'events': np.mean([item['score'] for item in result[2]]),
            'personality': np.mean([item['score'] for item in result[3]])
        })
        
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time needed: {elapsed_time}s")