
import json
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
import numpy as np
import os
import time
import re
import argparse
from utils import convert_json, convert_json_to_dict_golden

def gpt4_evaluator(dimension, character, predict_data, predict_data_gpt4, golden_data):
    prompt = eval_prompt.format(
        character=character,
        dimension=dimension,
        golden = golden_data,
        output_1 = predict_data,
        output_2 = predict_data_gpt4
    )
    response = evaluator.invoke(prompt).content
    
    try:
        result = eval(response)
        return result
    except:
        start_match = re.search(r'[\{]', response)
        # 查找从后面开始的最后一个']'或'}'
        end_match = re.search(r'[\}]', response[::-1])
        result = response
        if start_match and end_match:
            # 计算结束匹配的实际位置（因为我们是在反转的字符串中查找的）
            end_index = len(response) - end_match.start()
            # 提取并返回结果
            result = response[start_match.start():end_index]
        elif start_match and not end_match:
            if response[start_match.start()] == '[':
                # 字符串最后append]
                result = response[start_match.start():] + ']'
            else:
                result = response[start_match.start():] + '}'
            
        try:
            result = json.loads(result)
        except:
            print("Failed to evaluate:", response)
            return response
        
def evaluation(predict_data, golden_data, predict_data_gpt4):
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
        predict_trait_gpt4 = predict_data_gpt4[title]['traits']
        golden_trait = golden_data[title]['traits']
        traits_score = gpt4_evaluator('traits', character, predict_trait, predict_trait_gpt4, golden_trait)
        traits_scores.append(traits_score)
        
        predict_relationships = predict_data[title]['relationships']
        predict_relationships_gpt4 = predict_data_gpt4[title]['relationships']
        golden_relationships = golden_data[title]['relationships']
        relationship_score = gpt4_evaluator('relationships', character, predict_relationships, predict_relationships_gpt4, golden_relationships)
        relationships_scores.append(relationship_score)
        
        predict_events = predict_data[title]['events']
        predict_events_gpt4 = predict_data_gpt4[title]['events']
        golden_events = golden_data[title]['events']
        events_score = gpt4_evaluator('events', character, predict_events, predict_events_gpt4, golden_events)
        events_scores.append(events_score)
        
        predict_personality = predict_data[title]['personality']
        predict_personality_gpt4 = predict_data_gpt4[title]['personality']
        golden_personality = golden_data[title]['personality']
        personality_score = gpt4_evaluator('personality', character, predict_personality, predict_personality_gpt4, golden_personality)
        personality_scores.append(personality_score)
        
        gpt4_score[title] = {
        'traits': traits_score,
        'relationships': relationship_score,
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
    parser.add_argument("--predict_path", type=str, default="../exp/incremental/gpt-3.5-0125/result.json", help="path to the result of one model")
    parser.add_argument("--predict_path_gpt4", type=str, default="../exp/incremental/gpt-4-0125/result.json", help="path to the result of GPT-4")
    parser.add_argument("--golden_path", type=str, default = "../data/truth_persona_all_dimension.json", help="path to the json file storing the golden reference profile")
    parser.add_argument("--type", type=str, default="incremental", help="summarization method(incremental/hierarchical/once)")
    args = parser.parse_args()
    
    pid = os.getpid()
    print("pid:", pid)
    predict_path = args.predict_path
    golden_path = args.golden_path
    predict_path_gpt4 = args.predict_path_gpt4

    dir_path = os.path.dirname(predict_path)
    new_file_path = os.path.join(dir_path, "eval_win_rate.json")

    with open(predict_path, "r", encoding='utf-8') as f:
        predict_data = json.load(f)
    with open(predict_path_gpt4, "r", encoding='utf-8') as f:
        predict_data_gpt4 = json.load(f)
    with open(golden_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
    dir_path = os.path.dirname(predict_path)
        
    eval_prompt = open("./prompts/winwin_rate/prompt.txt", "r").read()
    
    predict_data = convert_json(predict_data, args.type)
    predict_data_gpt4 = convert_json(predict_data_gpt4, args.type)
    golden_data = convert_json_to_dict_golden(golden_data)
    
    evaluator = ChatOpenAI(model='gpt-4', temperature=0)

    start_time = time.time()

    with get_openai_callback() as cb:
        result = evaluation(predict_data, golden_data, predict_data_gpt4)

    print("predict_path", predict_path)
    print({
            'traits': np.mean([1 if result['model_name'] == 'model_1' else 0.5 if result['model_name'] == 'Equilibrium' else 0 for result in result[0]]),
            'relationships': np.mean([1 if result['model_name'] == 'model_1' else 0.5 if result['model_name'] == 'Equilibrium' else 0 for result in result[1]]),
            'events': np.mean([1 if result['model_name'] == 'model_1' else 0.5 if result['model_name'] == 'Equilibrium' else 0 for result in result[2]]),
            'personality': np.mean([1 if result['model_name'] == 'model_1' else 0.5 if result['model_name'] == 'Equilibrium' else 0 for result in result[3]]),
        })
        
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time needed: {elapsed_time}s")