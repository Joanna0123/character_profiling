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
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
import os
import time
import argparse
import json
import math
from collections import defaultdict
from utils import read_json, get_response, count_tokens

def check_summary_validity(summary, token_limit):
    if len(summary) == 0:
        raise ValueError("Empty summary returned")
    if count_tokens(summary) > token_limit or summary[-1] not in ['.', '?', '!', '\"', '\'']:
        return False
    else:
        return True

def summarize(character, texts, token_limit, level):
    text = texts['text']
    context = texts['context']
    word_limit = token_limit
    if level == 0:
        prompt = init_template.format(text, character, word_limit, word_limit-150)
    else:
        prompt = template.format(character, text, character, word_limit, word_limit-150)
        if len(context) > 0 and level > 0:
            prompt = context_template.format(character, context, character, text, character, word_limit, word_limit-150)
    print(f"PROMPT:\n\n---\n\n{prompt}\n\n---\n\n")
    response = get_response(prompt, temperature=0)
    print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")
    if level == 0 and response.startswith("I'm sorry"):
        feedback_prompt = init_feedback.format(character)
        print(f"\n\nINIT FEEDBACK PROMPT:\n{feedback_prompt}\n\n")
        response = get_response([prompt, response, feedback_prompt], temperature=0) 
        print(f"CHUNK SUMMARY: {response}\n")
    while len(response) == 0:
        print("Empty summary, retrying in 10 seconds...")
        time.sleep(10)
        print(f"PROMPT:\n\n---\n\n{prompt}\n\n---\n\n")
        response = get_response(prompt, temperature=0)
        print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")

    attempts = 0
    while not check_summary_validity(response, token_limit):
        word_limit = word_limit * (1 - 0.1 * attempts)
        if level == 0:
            prompt = init_template.format(text, character, word_limit, word_limit-150)
        else:
            prompt = template.format(character, text, character, word_limit, word_limit-150)
            if len(context) > 0 and level > 0:
                prompt = context_template.format(character, context, character, text, character, word_limit, word_limit-150)
        if attempts == 6:
            print("Failed to generate valid summary after 6 attempts, skipping")
            return response
        print(f"PROMPT:\n\n---\n\n{prompt}\n\n---\n\n")
        print(f"Invalid summary, retrying: attempt {attempts}")
        response = get_response(prompt, temperature=0)
        print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")
        attempts += 1
    return response


def estimate_levels(book_chunks, summary_limit=450):
    num_chunks = len(book_chunks)
    chunk_limit = CHUNK_SIZE
    levels = 0

    while num_chunks > 1:
        chunks_that_fit = (MAX_CONTEXT_LEN - count_tokens(template.format('', '', '', 0, 0)) - 20) // chunk_limit  # number of chunks that could fit into the current context
        num_chunks = math.ceil(num_chunks / chunks_that_fit)  # number of chunks after merging
        chunk_limit = summary_limit
        levels += 1

    summary_limits = [MAX_SUMMARY_LEN]
    for _ in range(levels-1):
        summary_limits.append(int(summary_limits[-1] * WORD_RATIO))
    summary_limits.reverse()  # since we got the limits from highest to lowest, but we need them from lowest to highest
    return levels, summary_limits

def recursive_summary(book, character, summaries, level, chunks, summary_limits):
    """
    Merges chunks into summaries recursively until the summaries are small enough to be summarized in one go.

    chunks: list of chunks
    level: current level
    summaries_dict: dictionary of summaries for each level
    summary_limits: list of summary limits for each level
    """
    print(f"Level {level} has {len(chunks)} chunks")
    i = 0
    if level == 0 and len(summaries[book]['summaries_dict'][0]) > 0:
        # resume from the last chunk
        i = len(summaries[book]['summaries_dict'][0])
    if level >= len(summary_limits):  # account for underestimates
        summary_limit = MAX_SUMMARY_LEN
    else:
        summary_limit = summary_limits[level]
    
    summaries_dict = summaries[book]['summaries_dict']

    if level > 0 and len(summaries_dict[level]) > 0:
        if count_tokens('\n\n'.join(chunks)) + MAX_SUMMARY_LEN + count_tokens(context_template.format('', '', '', '', '', 0, 0)) + 20 <= MAX_CONTEXT_LEN:  # account for overestimates
            summary_limit = MAX_SUMMARY_LEN
        num_tokens = MAX_CONTEXT_LEN - summary_limit - count_tokens(context_template.format('','', '', '', '', 0, 0)) - 20  # Number of tokens left for context + concat
    else:
        if count_tokens('\n\n'.join(chunks)) + MAX_SUMMARY_LEN + count_tokens(template.format('', '', '', 0, 0)) + 20 <= MAX_CONTEXT_LEN:
            summary_limit = MAX_SUMMARY_LEN
        num_tokens = MAX_CONTEXT_LEN - summary_limit - count_tokens(template.format('', '', '', 0, 0)) - 20

    while i < len(chunks):
        context = ""
        # Generate previous level context
        context = summaries_dict[level][-1] if len(summaries_dict[level]) > 0 else ""

        texts = {}
        # Concatenate as many chunks as possible
        if level == 0:
            text = chunks[i]
        else:
            j = 1
            text = f"Summary {j}:\n\n{chunks[i]}"
            while i + 1 < len(chunks) and count_tokens(context + text + f"\n\nSummary {j+1}:\n\n{chunks[i+1]}") + 20 <= num_tokens:
                i += 1
                j += 1
                text += f"\n\nSummary {j}:\n\n{chunks[i]}"
        texts = {
            'text': text,
            'context': context
        }

        # Calling the summarize function to produce the summaries
        print(f"Level {level} chunk {i}")
        print(f"Summary limit: {summary_limit}")
        summary = summarize(character, texts, summary_limit, level)
        summaries_dict[level].append(summary)
        i += 1

        # json.dump(summaries, open(SAVE_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    # If the summaries still too large, recursively call the function for the next level
    if len(summaries_dict[level]) > 1:
        # save the current level summaries
        return recursive_summary(book, character, summaries, level + 1, summaries_dict[level], summary_limits)
    else:
        return summaries_dict[level][0]  # the final summary


def summarize_book(book, chunks, character, summaries):
    levels, summary_limits = estimate_levels(chunks)
    print(f"Book {book} has {levels} levels by estimate")
    print(f"Summary limits: {summary_limits}")
    
    level = 0
    if len(summaries[book]['summaries_dict']) > 0:
        if len(summaries[book]['summaries_dict']) == 1:  # if there is only one level so far
            if len(summaries[book]['summaries_dict'][0]) == len(chunks):  # if level 0 is finished, set level to 1
                level = 1
            elif len(summaries[book]['summaries_dict'][0]) < len(chunks):  # else, resume at level 0
                level = 0
            else:
                raise ValueError(f"Invalid summaries_dict at level 0 for {book}")
        else:  # if there're more than one level so far, resume at the last level
            level = len(summaries[book]['summaries_dict'])
        print(f"Resuming at level {level}")
    
    final_summary = recursive_summary(book, character, summaries, level, chunks, summary_limits)
    
    return final_summary, summaries


def get_hierarchical_summaries(book_data, summaries):
    book = book_data['title']
    src = f'../data/books/{book}/{book}_{CHUNK_SIZE}.json'
    with open(src, 'r', encoding="utf-8") as f:
        split_book = json.load(f)
        character = list(book_data['persona'].keys())[0]
    book_chunk = [chunk["text"] for chunk in split_book]
    

    if book in summaries and 'final_summary' in summaries[book]:
        print("Already processed, skipping...")
        return
        
    chunks = book_chunk
    if book in summaries and 'summaries_dict' in summaries[book]:
        if len(summaries[book]['summaries_dict']) == 1 and len(summaries[book]['summaries_dict'][0]) < len(chunks):
            level = 0
        elif len(summaries[book]['summaries_dict']) == 1 and len(summaries[book]['summaries_dict'][0]) == len(chunks):
            level = len(summaries[book]['summaries_dict']) - 1
            chunks = summaries[book]['summaries_dict'][level]
    else:
        summaries[book] = {
            'summaries_dict': defaultdict(list)
        }
    final_summary, summaries = summarize_book(book, chunks, character, summaries)
    summaries[book]['final_summary'] = final_summary
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=4)
            
def main(raw_data, summaries):

    for i, book in enumerate(raw_data):
        get_hierarchical_summaries(book, summaries)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="../exp/hierarchical/gpt-4-0125/", help="path to the json file to save the result")
    parser.add_argument("--max_context_len", type=int, default = 8096, help="max content length of the model")
    parser.add_argument("--chunk_size", type=int, default=3000)
    parser.add_argument("--max_summary_len", type=int, default=1200, help="max length of the final summary")
    parser.add_argument("--prompt", type=str, default="profile_hierarchical", help="prompt file name")
    args = parser.parse_args()
    
    pid = os.getpid()
    print("pid:", pid)
    dir = "../data/supersummary.json"
    _, raw_data = read_json(dir)
    print(len(raw_data))
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    SAVE_PATH = os.path.join(args.save_folder, 'result.json')
    CHUNK_SIZE = args.chunk_size
    MAX_CONTEXT_LEN = args.max_context_len
    MAX_SUMMARY_LEN = args.max_summary_len
    WORD_RATIO = 0.65

    init_template = open(f"./prompts/{args.prompt}/init.txt", "r").read()
    template = open(f"./prompts/{args.prompt}/merge.txt", "r").read()
    context_template = open(f"./prompts/{args.prompt}/merge_context.txt", "r").read()
    init_feedback = "If there is no information about character {} in this part of the story, just output 'None' in each section. Do not apologize. Just output in the required format."
    
    
    summaries = defaultdict(dict)
    if os.path.exists(SAVE_PATH):
        print("Loading existing summaries...")
        summaries = json.load(open(SAVE_PATH, 'r',encoding='utf-8'))
        # convert all keys into int
        for book in summaries:
            summaries[book]['summaries_dict'] = defaultdict(list, {int(k): v for k, v in summaries[book]['summaries_dict'].items()})
            
    start_time = time.time()
    with get_openai_callback() as cb:
        
        main(raw_data, summaries)
        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time needed: {elapsed_time}s")