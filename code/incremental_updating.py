# Copyright (c) 2024 Yapei Chang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following code has been modified based on a project licensed under the MIT License.
# Original code source: [https://github.com/lilakk/BooookScore]

from langchain.callbacks import get_openai_callback
import tqdm

from utils import read_json, count_tokens, get_response
import argparse
import json
import os

import time
   
def compress(character, response, summary, chunk, summary_len, word_limit, num_chunks, j, prompts):
    intermediate_summary_prompt = prompts['update']
    compression_prompt = prompts['compress']
    chunk_trims = 0
    compressed_summary = None
    summary_words = len(summary.split())
    ori_expected_words = max(int(word_limit * j / num_chunks), 300)  # no need to be j + 1 since we're compressing the summary at the previous chunk
    # ori_expected_words = word_limit
    expected_words = ori_expected_words
    actual_words = expected_words

    dic = {}  # keep track of each trimmed summary and their actual number of words

    while response[-1] not in ['.', '?', '!', '\"', '\''] \
    or count_tokens(response) >= summary_len or actual_words > int(expected_words * 1.2): # or actual_words < int(expected_words * 0.8)
        if chunk_trims == 6:
            print(f"\nCOMPRESSION FAILED AFTER 6 ATTEMPTS, SKIPPING\n")
            if not all([v['valid_response'] == False for v in dic.values()]):
                dic = {k: v for k, v in dic.items() if v['valid_response'] == True}
            print(f"DICTIONARY LENGTH: {len(dic)}")
            closest_key = min(dic, key=lambda x:abs(x-ori_expected_words))  # find the trimmed summary with actual # words closest to the expected # words
            print(f"EXPECTED WORDS: {ori_expected_words} | CLOSEST KEY: {closest_key} | ALL KEYS: {dic.keys()}")

            return dic[closest_key]['compressed_summary'], dic[closest_key]['response'], chunk_trims, 1
        
        compressed_summary = summary
        if count_tokens(summary) > expected_words:
            print(f"\nCOMPRESSION REQUIRED, ATTEMPT {chunk_trims + 1}\n")
            print(f"MAX LEN: {summary_len} | ACTUAL LEN: {count_tokens(response)}")

            expected_words = max(int(ori_expected_words * (1 - chunk_trims * 0.05)), 300)
            prompt = compression_prompt.format(character, summary, summary_words, expected_words,  expected_words-150, expected_words)
            response = get_response(prompt, temperature=0)
            compressed_summary = response
            print(f"\n\nPROMPT:\n{prompt}\n\n")
            print(f"TRIMMED SUMMARY: {compressed_summary}\n")
            actual_words = len(compressed_summary.split())
            current_tokens = count_tokens(compressed_summary)
            print(f"EXPECTED WORDS: {expected_words} | ACTUAL WORDS: {actual_words} | CURRENT TOKENS: {current_tokens}\n\n")
            print("-" * 10)
            if compressed_summary[-1] not in ['.', '?', '!', '\"', '\''] \
            or count_tokens(compressed_summary) >= summary_len \
            or actual_words > int(expected_words * 1.2): # or actual_words < int(expected_words * 0.8) 
                print(f"INVALID TRIMMED SUMMARY, CONTINUE TO NEXT ATTEMPT\n\n")
                chunk_trims += 1
                if chunk_trims < 6:
                    continue
                else:
                    print(f"\nCOMPRESSION FAILED AFTER 6 ATTEMPTS, SKIPPING\n")
        
        num_words = max(int(word_limit * (j + 1) / num_chunks), 300)
        prompt = intermediate_summary_prompt.format(chunk, character, compressed_summary, character, num_words)
        print(f"\n\nPROMPT:\n{prompt}\n\n")
        response = get_response(prompt, temperature=0)
        print(f"UPDATED TRIMMED SUMMARY: {response}\n")
        if response.startswith("I'm sorry"):
            feedback_prompt = update_feedback.format(character, character)
            print(f"\n\nFEEDBACK PROMPT:\n{feedback_prompt}\n\n")
            response = get_response([prompt, response, feedback_prompt], temperature=0) 
            print(f"UPDATED TRIMMED SUMMARY: {response}\n")
        dic[actual_words] = {
            'compressed_summary': compressed_summary,
            'response': response,
            'valid_response': response[-1] in ['.', '?', '!', '\"', '\''] \
            and count_tokens(response) < summary_len
        }
        print(f"\n\nPROMPT:\n{prompt}\n\n")
        print(f"VALID_RESPONSE: {dic[actual_words]['valid_response']}")
        print("-" * 10)
        if chunk_trims == 6:
            break
        chunk_trims += 1

    return compressed_summary, response, chunk_trims, 0

def process_book(book_data, book_index, prompts):
    
    initial_summary_prompt = prompts['init']
    intermediate_summary_prompt = prompts['update']
    
    book_name = book_data['title']
    src = f'../data/books/{book_name}/{book_name}_{CHUNK_SIZE}.json'
    with open(src, 'r', encoding="utf-8") as f:
        split_book = json.load(f)
        character = list(book_data['persona'].keys())[0]
    book_chunk = [chunk["text"] for chunk in split_book]

    if book_name in new_data and len(new_data[book_name]) >= len(book_chunk):
        print(f"Skipping {book_data} because it already exists in {SAVE_PATH}")
        return

    new_chunks = []
    prev_summary = None
    if len(new_data) > book_index and book_name in new_data:
        new_chunks = new_data[book_name]
        prev_summary = new_chunks[-1]
    dd = book_chunk
    summary_len = MAX_SUMMARY_LEN
    word_limit = int(summary_len * WORD_RATIO)
    num_chunks = len(dd)

    for j, chunk in tqdm.tqdm(enumerate(book_chunk)):
        if j < len(new_chunks):
                print(f"Skipping chunk {j}...")
                continue

        if prev_summary is None:
            prompt = initial_summary_prompt.format(chunk, character, summary_len)
        else:
            prompt = intermediate_summary_prompt.format(chunk, character, prev_summary, character, summary_len)
        
        response = get_response(prompt,temperature=0)
        print(f"\n\nPROMPT:\n{prompt}\n\n")
        print(f"\n\nCHUNK SUMMARY:\n{response}\n\n")
        
        if prev_summary is None and response.startswith("I'm sorry"):
            feedback_prompt = init_feedback.format(character)
            print(f"\n\nINIT FEEDBACK PROMPT:\n{feedback_prompt}\n\n")
            response = get_response([prompt, response, feedback_prompt], temperature=0) 
            print(f"CHUNK SUMMARY: {response}\n")
        elif prev_summary is not None and response.startswith("I'm sorry"):
            feedback_prompt = update_feedback.format(character, character)
            print(f"\n\nUPDATE FEEDBACK PROMPT:\n{feedback_prompt}\n\n")
            response = get_response([prompt, response, feedback_prompt], temperature=0) 
            print(f"CHUNK SUMMARY: {response}\n")
        if not response:
            print("\n\nNo response. Retry in 30 seconds.\n\n")
            time.sleep(30)
            response = get_response(prompt, temperature=0)
        
        actual_words = len(response.split())
        print(f"ACTUAL WORDS: {actual_words}")
        print("-" * 10)
        
        # compress prev_summary if the current one is too long or doesn't end in punctuation
        if prev_summary is not None and (response[-1] not in ['.', '?', '!', '\"', '\''] \
        or count_tokens(response) >= summary_len):
            compressed_summary, response, chunk_trims, skipped = compress(character, response, prev_summary, chunk, summary_len, word_limit, num_chunks, j, prompts)
            new_chunks[j - 1] = compressed_summary

        prev_summary = response
        new_chunks.append(response)

    new_data[book_name] = new_chunks
    print(f"Saving data for book {book_index}...")
    json.dump(new_data, open(SAVE_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

def main(raw_data, prompts):

    for i, book in enumerate(raw_data):
        process_book(book, i, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="../exp/incremental/gpt-4-0125/", help="path to the json file to save the data")
    parser.add_argument("--chunk_size", type=int, default=3000)
    parser.add_argument("--max_summary_len", type=int, default=1200, help="max length of the final summary")
    parser.add_argument("--prompt", type=str, default="profile_incremental", help="prompt file name")
    args = parser.parse_args()

    pid = os.getpid()
    print("pid:", pid)
    dir = "../data/supersummary.json"
    _, raw_data = read_json(dir)
    print(len(raw_data))
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    SAVE_PATH = os.path.join(args.save_folder, 'result.json')
    MAX_SUMMARY_LEN = args.max_summary_len
    CHUNK_SIZE = args.chunk_size
    WORD_RATIO = 0.65
    
    with open(f"./prompts/{args.prompt}/init.txt", "r") as f:
        init_template = f.read()
    with open(f"./prompts/{args.prompt}/update.txt", "r") as f:
        update_template = f.read()
    with open(f"./prompts/{args.prompt}/compress.txt", "r") as f:
        compress_template = f.read()
    prompts = {
        'init': init_template,
        'update': update_template,
        'compress': compress_template
    }
    update_feedback = "If there is no information about character {} in this excerpt, just output the origin summary of the character {} of the story up until this point. Do not apologize. Just output in the required format."
    init_feedback = "If there is no information about character {} in the beginning part of a story, just output 'None' in each section. Do not apologize. Just output in the required format."
        
    if os.path.exists(SAVE_PATH):
        new_data = json.load(open(SAVE_PATH, "r", encoding='utf-8'))
    else:
        new_data= {}
        
    with get_openai_callback() as cb:
        start_time = time.time()
        
        main(raw_data, prompts)
        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time needed: {elapsed_time} 秒")