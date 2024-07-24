from langchain.callbacks import get_openai_callback
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
import os
import traceback
import time
import argparse
import json
from collections import defaultdict
from utils import read_json, once_titles
import anthropic
from openai import OpenAI

def get_response(prompt, model):
    
    response = None
    num_attemps = 0
    while response is None and num_attemps < 3:
        try:
            if model=='gpt4':
                client = OpenAI(
                    max_retries=3,
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-4-0125-preview",
                    temperature=0,
                    max_tokens=4096,
                )
                response = chat_completion.choices[0].message.content
            elif model=='claude3':
                client = anthropic.Anthropic(
                    api_key = os.environ["ANTHROPIC_API_KEY"]
                )
                message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = message.content
        except Exception as e:
            print(traceback.format_exc())
            num_attemps += 1
            print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
            time.sleep(5)
            
    return response

# summarize in one go
def summarize_book(chunks, character, model):
    prompt = template.format(character, MAX_SUMMARY_LEN, chunks[0])
    response = get_response(prompt, model)
    return response
    
def get_once_summaries(book_data, model):
    book = book_data['title']
    src = f'../data/books/{book}/{book}_{CHUNK_SIZE}.json'
    with open(src, 'r', encoding="utf-8") as f:
        split_book = json.load(f)
        character = list(book_data['persona'].keys())[0]
    book_chunk = [chunk["text"] for chunk in split_book]

    chunks = book_chunk
    
    final_summary = summarize_book(chunks, character, model)
    summaries[book] = {'final_summary': final_summary}
          
def main(raw_data, model):
    for _, book in enumerate(raw_data):
        if book['title'] not in once_titles:
            continue
        if book['title'] not in summaries:
            get_once_summaries(book, model)
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="../exp/once/gpt-4-0125/", help="save folder")
    parser.add_argument("--chunk_size", type=int, default=120000)
    parser.add_argument("--max_summary_len", type=int, default=1200, help="max length of the final summary")
    parser.add_argument("--model", type=str, default="gpt4")
    parser.add_argument("--prompt", type=str, default="profile_once", help="prompt file name")
    args = parser.parse_args()
    
    pid = os.getpid()
    print("pid:", pid)
    
    _, raw_data = read_json()
    
    # print(len(raw_data))
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    SAVE_PATH = os.path.join(args.save_folder, 'result.json')
    CHUNK_SIZE = args.chunk_size
    MAX_SUMMARY_LEN = args.max_summary_len

    template = open(f"./prompts/{args.prompt}/prompt.txt", "r").read()
    
    summaries = defaultdict(dict)
    if os.path.exists(SAVE_PATH):
        print("Loading existing summaries...")
        summaries = json.load(open(SAVE_PATH, 'r'))
        
    start_time = time.time()
    with get_openai_callback() as cb:
        
        main(raw_data, args.model)
        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time needed: {elapsed_time}s")