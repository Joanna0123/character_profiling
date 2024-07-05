import os
import json
from tqdm import tqdm
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from utils import convert_json, get_response
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import numpy as np


def main(type, test_result_file, ablation):
    with open(multi_choice_questions_file, "r", encoding="utf-8") as f, open(
        persona_file, "r", encoding="utf-8"
    ) as g, open(test_result_file, "a", encoding="utf-8") as h:
        multi_choice_questions = json.load(f)
        persona = json.load(g)
        persona = convert_json(persona, type)
        answers = {}
        prompt_list = []
        titles = list(persona.keys())
        if type == "once":
            assert len(titles) == 47
        else:
            assert len(titles) == 126
        id_list = []
        for book_title in titles:
            for character, questions in multi_choice_questions[book_title].items():
                for question in questions:
                    option = "\n".join(question["Multiple Choice Question"]["Options"])
                    question_format = f"""Scenario: {question["Multiple Choice Question"]["Scenario"]}\nQuestion: {question["Multiple Choice Question"]["Question"]}\nOptions:\n{option}"""

                    id = question["Multiple Choice Question"]["id"]
                    
                    if type == "incremental" or type == "hierarchical" or type == "once" or type == "golden":
                        character_persona = persona[book_title]
                        summary = character_persona['summary']
                        if not ablation:
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                summary,
                                question_format
                            )
                        # for ablation study
                        elif ablation == "no_traits":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['relationships'],character_persona['events'], character_persona['personality']]),
                                question_format
                            )
                        elif ablation == "no_relationships":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['traits'],character_persona['events'], character_persona['personality']]),
                                question_format
                            )
                        elif ablation == "no_events":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['traits'],character_persona['relationships'], character_persona['personality']]),
                                question_format
                            )
                        elif ablation == "no_personality":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['traits'],character_persona['relationships'], character_persona['events']]),
                                question_format
                            )
                        elif ablation == "no_tr_re":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['events'], character_persona['personality']]),
                                question_format
                            )
                        elif ablation == "no_tr_re_ev":
                            prompt = PROFILE_BASELINE.format(
                                character,
                                character,
                                ''.join([character_persona['personality']]),
                                question_format
                            )
                        elif ablation == 'none_profile':
                            prompt = PROFILE_ANALYSIS.format(
                                character,
                                character,
                                question_format
                            )
                    prompt_list.append(prompt)
                    id_list.append(id)
        try:
            responses = []
            # batch_size 50
            for i in tqdm(range(0, len(prompt_list), 50)):
                batch = prompt_list[i:i+50]
                batch_responses = client.batch([[HumanMessage(content=prompt),] for prompt in batch])
                responses.extend(batch_responses)
            # for prompt in tqdm(prompt_list):
            #     responses.append(get_response(prompt))
            # match response with id
            for response, id in zip(responses, id_list):
                answers[str(id)] = eval(response.content)
        except Exception as e:
            print(f"sth is wrong with {book_title}")
        json.dump(answers, h, ensure_ascii=False)
    return answers

def eval_motivation(answers):
    truth_dir = multi_choice_questions_file
    with open(truth_dir, "r", encoding='utf-8') as f:
        truth = json.load(f)
    transformed_truth = {}

    for book_title, personas in truth.items():

        for persona_name, questions in personas.items():
            for question_dict in questions:
                question_info = question_dict["Multiple Choice Question"]
                if "Correct Answer" not in question_info:
                    print(question_info)
                question_id = str(question_info["id"])
                correct_answer = question_info["Correct Answer"]
                transformed_truth[question_id] = correct_answer
                
    accuracies = []
    for i, answer in enumerate(answers):
        question_accuracy = {question_id: False for question_id in answer.keys()}

        for question_id, res_data in answer.items():
            if question_id in transformed_truth:
                correct_answer = transformed_truth[question_id]
                if isinstance(res_data, str):
                    res_data = eval(res_data)
                if res_data['Choice'] == correct_answer:
                    question_accuracy[question_id] = True
                    
        all_wrong_question_ids = [question_id for question_id, is_correct in question_accuracy.items() if not is_correct]
        accuracy = (len(answer)- len(all_wrong_question_ids))/len(answer)
        accuracies.append(accuracy)
        print(f"test {i+1} accuracy {accuracy}")
        print(f"wrong IDs:{len(all_wrong_question_ids)}: {all_wrong_question_ids}")
    numbers_np = np.array(accuracies)
    mean = np.mean(numbers_np)
    variance = np.std(numbers_np)
    print(f"accuracy mean: {mean}，std: {variance}")
        
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Motivation Recognition')
    parser.add_argument('--persona_file', type=str, default="../exp/incremental/gpt-4-0125/result.json")
    parser.add_argument('--type', type=str, default='incremental', help='summarization method(incremental/hierarchical/once/golden)')
    parser.add_argument('--num_attempts', type=int, default=1, help='number of attempts')
    parser.add_argument('--ablation', type=str, default=None, help='ablation type(None/no_traits/no_relationships/no_events/no_personality/no_tr_re/no_tr_re_ev/none_profile)')
    args = parser.parse_args()
    pid = os.getpid()
    print("pid:", pid)

    persona_file = args.persona_file
    multi_choice_questions_file = "../data/motivation_dataset_rewrite.json"

    dir_path = os.path.dirname(persona_file)
    
    PROFILE_BEST = open("./prompts/motivation_test/profile_best.txt", "r").read()
    PROFILE_BASELINE = open("./prompts/motivation_test/profile_golden.txt", "r").read()
    PROFILE_ANALYSIS = open("./prompts/motivation_test/profile_analysis.txt", "r").read()
    client = ChatOpenAI(model='gpt-4-0125-preview', temperature=0, response_format={ "type": "json_object" },)

    with get_openai_callback() as cb:
        print(f"profile for motivation recognition is {persona_file}")
        if args.type == "once":
            result_files = [f"motivation_score_test{i+1}_once.json" if not args.ablation else f"motivation_score_test{i+1}_{args.ablation}.json" for i in range(args.num_attempts)]
        else:
            result_files = [f"motivation_score_test{i+1}.json" if not args.ablation else f"motivation_score_test{i+1}_{args.ablation}.json" for i in range(args.num_attempts)]
            
        answers = []

        for result_file in result_files:
            test_result_file = os.path.join(dir_path, "motivation_score", result_file)
            if not os.path.exists(os.path.dirname(test_result_file)):
                os.makedirs(os.path.dirname(test_result_file))
            if not os.path.exists(test_result_file):
                answer = main(args.type, test_result_file, args.ablation)
            else:
                with open(test_result_file, "r", encoding='utf-8') as f:
                    answer = json.load(f)
                if len(answer) == 0:
                    answer = main(args.type, test_result_file, args.ablation)
            answers.append(answer)
            
        eval_motivation(answers)
        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    
