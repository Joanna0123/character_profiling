import json
from transformers import GPT2Tokenizer
import traceback
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import time
set_llm_cache(InMemoryCache())

def count_tokens(text):
    return len(text.split())

def get_response(prompt,  temperature=0):

    # gpt model
    llm = ChatOpenAI(model='gpt-4-0125-preview', temperature=temperature, max_tokens=4096)
    
    # vllm-serving model
    # llm = ChatOpenAI(
    #     openai_api_key="EMPTY", 
    #     openai_api_base=f"http://localhost:8001/v1", 
    #     model="path_to_model", 
    #     max_retries=3, 
    #     max_tokens=4096, 
    #     temperature = temperature,)
    
    response = None
    num_attemps = 0
    while response is None and num_attemps < 3:
        try:
            if isinstance(prompt, str):
                response = llm.invoke(
                    [HumanMessage(content = prompt),]
                ).content
            else:
                messages = []
                for i, message in enumerate(prompt):
                    if i%2 == 0:
                        messages.append(HumanMessage(content=message))
                    else:
                        messages.append(AIMessage(content=message))
                response = llm.invoke(
                    messages
                ).content

        except Exception as e:
            print(traceback.format_exc())
            num_attemps += 1
            print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
            time.sleep(5)
            
    return response

def read_json(dir: str):
    truth = '../data/truth_persona_all_dimension.json'
    with open(truth, "r") as f:
        truth = json.load(f)
    truth_title = [item['title'] for item in truth]

    SS_datasets = []
    SS_persona_data = []
    with open(dir, "r") as f:
        raw_data = json.load(f)
    for title, book in raw_data.items():
        if title not in truth_title:
            continue
        chapter_summaries = {}
        concat_summary = ""
        for chapter, summary in book['chapter_summaries'].items():
            concat_summary += summary
            if 'analysis' in chapter.lower():
                chapter_summaries[chapter] = {'summary':concat_summary,'analysis':summary}
                concat_summary = ""
        characters = {}
        character = next(iter(book['characters']))
        
        characters[character] = {}
        SS_datasets.append(
            {
                "title": title,
            }
        )
        SS_persona_data.append(
            {
                "title": title,
                "persona": characters,
            }
        )

    return SS_datasets, SS_persona_data


def parse_character_summary(text):
    # parse the summary into four sections
    section_titles = ["Attributes", "Relationships", "Events", "Personality"]
    sections = {title: "" for title in section_titles}
    
    current_section = None
    for line in text.split('\n'):
        for title in section_titles:
            if title in line:
                current_section = title  
                break

        if current_section:
            sections[current_section] += line + '\n'
            
    missing_sections = [title for title, content in sections.items() if not content]
    if missing_sections:
        print("Missing sections:", ", ".join(missing_sections))

    return sections

def convert_json(original_data, type):
    transformed_data = {}

    if type == "incremental":
        for title, summaries in original_data.items():

            sections = parse_character_summary(summaries[-1])
            transformed_data[title] = {
                'summary':summaries[-1],
                'traits' : sections.get("Attributes", ""),
                'relationships' : sections.get("Relationships", ""),
                'events' : sections.get("Events", ""),
                'personality' : sections.get("Personality", ""),
                }
    elif type == "hierarchical":
        for title, summaries in original_data.items():

            sections = parse_character_summary(summaries['final_summary'])
            transformed_data[title] = {
                'summary':summaries['final_summary'],
                'traits' : sections.get("Attributes", ""),
                'relationships' : sections.get("Relationships", ""),
                'events' : sections.get("Events", ""),
                'personality' : sections.get("Personality", ""),
                }
        print(len(transformed_data))
    elif type == "once":
        for title, summaries in original_data.items():
            if title not in once_titles:
                continue
            if isinstance(summaries, list):
                summary = summaries[-1]
            else:
                summary = summaries['final_summary']
            sections = parse_character_summary(summary)
            transformed_data[title] = {
                'summary': summary,
                'traits' : sections.get("Attributes", ""),
                'relationships' : sections.get("Relationships", ""),
                'events' : sections.get("Events", ""),
                'personality' : sections.get("Personality", ""),
                }
    if type == "once":
        assert len(transformed_data) == 47
    else:
        assert len(transformed_data) == 126
    return transformed_data

def convert_json_to_dict_golden(json_data):

    result_dict = {}
    for item in json_data:
        title = item.get('title')
        persona = item.get('persona', {})
        
        for name, details in persona.items():
            traits = details.get('traits', {})
            relationships = details.get('relationships', [])
            events = details.get('events', [])
            personality = details.get('personality', {})
            
            result_dict[title] = {
                'character': name,
                'traits': traits,
                'relationships': relationships,
                'events': events,
                'personality': personality
            }
    return result_dict


once_titles = ['All Good People Here', 'All My Rage', 'The Bodyguard', 'Carrie Soto Is Back', 'Counterfeit', 'The Club', 'Daisy Darker', 'Dreamland', 'Every Summer After', 'French Braid', 'Funny You Should Ask', 'How High We Go in the Dark', 'The House Across the Lake', 'The Housemaid', 'I Must Betray You', 'The Inmate', 'Killers of a Certain Age', 'Lapvona', 'Legends & Lattes', 'The Last to Vanish', 'The Lies I Tell', 'Nora Goes Off Script', 'Notes on an Execution', 'Odder', 'One Italian Summer', 'One of Us Is Dead', 'Other Birds', 'Reminders of Him', 'A Scatter Of Light', 'Sea of Tranquility', 'Signal Fires', 'The Swimmers', 'This Time Tomorrow', 'True Biz', 'Trust', 'Two Degrees', 'Upgrade', 'The Very Secret Society of Irregular Witches', 'Yellowface', 'Meet Me at the Lake', 'None of This Is True', 'Pineapple Street', 'River Sing Me Home', 'Romantic Comedy', 'Small Mercies', 'The House in the Pines', "The Housemaid's Secret"]