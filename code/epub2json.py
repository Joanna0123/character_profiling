# The functions `process_epub`, `remove_blank_line`, `divide_str`, `strong_divide`, `split_book` are adapted from a project licensed under the Apache License, Version 2.0.
# Original code source: [https://github.com/LC1332/Chat-Haruhi-Suzumiya]
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ebooklib
from ebooklib import epub
import json
from bs4 import BeautifulSoup
import os
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

from utils import once_titles

# transform epub files into txt files
def process_epub(epub_file, txt_file):
  book = epub.read_epub(epub_file)
  with open(txt_file,"w",encoding="utf-8") as f:
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'xml')
            f.write(soup.text)

# remove blank lines in txt files
def remove_blank_line(src,dest):
  with open(src,'r',encoding='utf-8') as r, open(dest,'w',encoding='utf-8') as w:
    for text in r.readlines():
        if text.split():
            w.write(text)

def divide_str(s, sep=['\n', '.', '!','?']):
    mid_len = len(s) // 2 
    best_sep_pos = len(s) + 1 
    best_sep = None 
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos -
                                                        mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]

def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',\
                 '：', '!', '?', '(', ')', '”', '“', \
                 '’', '‘', '[', ']', '{', '}', '<', '>', \
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''#''', '@', '&', '^', '_', '`', '~',\
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]

def split_book(raw_text,max_token_len = 1500):
    chunk_text = []
    split_text = raw_text.split('\n')

    curr_len = 0
    curr_chunk = ''

    tmp = []

    for line in split_text:
        line = line.strip('\n ')
        line_len = len(enc.encode( line ))

        if line_len <= max_token_len - 5:
            tmp.append(line)
        else:
            # print('divide line with length = ', line_len)
            path = [line]
            tmp_res = []

            while path:
                my_str = path.pop()
                left, right = strong_divide(my_str)

                len_left = len(enc.encode( left ))
                len_right = len(enc.encode( right ))

                if len_left > max_token_len - 15:
                    path.append(left)
                else:
                    tmp_res.append(left)

                if len_right > max_token_len - 15:
                    path.append(right)
                else:
                    tmp_res.append(right)

            for line in tmp_res:
                tmp.append(line)

    split_text = tmp

    for line in split_text:
        line = line.strip('\n ')
        line_len = len(enc.encode(line))

        if line_len > max_token_len:
            print('warning line_len = ', line_len)

        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len

    if curr_chunk:
        chunk_text.append(curr_chunk)

    # 
    if len(chunk_text[-1]) < 1000:
        chunk_text[-2] += chunk_text[-1]
        temp = chunk_text[-1]
        chunk_text.remove(temp)
    return chunk_text

def write_txt(book_names):
    for book_name in book_names:
        epub_file = "../data/books/epub/"+book_name+".epub"
        txt_file = "../data/books/"+book_name+"/"+"raw_"+book_name+".txt"
        folder = os.path.exists("../data/books/"+book_name+"/")
        if not folder:
            os.makedirs("../data/books/"+book_name+"/")
        if os.path.exists(txt_file):
            continue
        else:
            process_epub(epub_file,txt_file)
        
def write_json(book_names, chunk_size=3000):
    for book_name in book_names:
        txt_file = "../data/books/"+book_name+"/"+"raw_"+book_name+".txt"
        processed_txt_file = "../data/books/"+book_name+"/"+book_name+".txt"
        
        remove_blank_line(txt_file,processed_txt_file)
        raw_text = open(processed_txt_file, encoding='utf-8').read()
        json_name = "../data/books/"+book_name+"/"+book_name+f"_{chunk_size}.json"
        if os.path.exists(json_name):
            continue
        chunk_text = split_book(raw_text,chunk_size)
        output = []
        for i,chunk in enumerate(chunk_text):
            output.append({'Number':i+1,'text':chunk})
        with open(json_name, 'w',encoding='utf-8') as file:
            json.dump(output, file,ensure_ascii=False)
    
if __name__ == "__main__":
    with open('../data/truth_persona_all_dimension.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    book_names = [book['title'] for book in data]
    write_json(book_names, 3000)
    write_json(once_titles, 120000)