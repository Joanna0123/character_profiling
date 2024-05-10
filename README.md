## Preprocess
Due to copyright reasons, we will not publicly release the original text of the books. We recommend that researchers purchase the ePub book version recommended on SuperSummary and place it in the `data/books/epub` folder, with the filename being the title of the book. Then process the original ePub files into JSON files using the following code for experiments.

```bash
cd code
python epub2json.py
```

## Generate Character Profiles
Below are example command lines for generating character profiles:

```bash
cd code
# incremental updating
python incremental_updating.py --save_folder ../exp/incremental/gpt-4-0125/ --chunk_size 3000 --max_summary_len 1200 --prompt profile_incremental

# hierarchical summarizing
python hierarchical_summarizing.py --save_folder ../exp/hierarchical/gpt-4-0125/ --chunk_size 3000 --max_summary_len 1200 --prompt profile_hierarchical

# summarize in one go
python summarize_in_one_go.py --save_folder ../exp/once/gpt-4-0125/ --chunk_size 120000 --max_summary_len 1200 --model gpt4 --prompt profile_once
```

## Factual Consistency Examination
Due to copyright reasons, we are temporarily not releasing golden reference character profiles. If you need access to the data for academic research purposes, please contact us via email.(See "Contact" section below)

Below are example command lines for factual consistency examination:

```bash
# The 'type' parameter can be one of the following: 'incremental', 'hierarchical', 'once'.

cd code
# consistency score
python evaluation_score.py --predict_path ../exp/incremental/gpt-4-0125/result.json --golden_path ../data/truth_persona_all_dimension.json --type incremental

# win-win rate
python evaluation_win.py --predict_path ../exp/incremental/gpt-3.5-0125/result.json --predict_path_gpt4 ../exp/incremental/gpt-4-0125/result.json --golden_path ../data/truth_persona_all_dimension.json --type incremental
```

## Motivation Recognition
The dataset for motivation recognition task is located at `data/motivation_dataset.json`.
Here's the format of the JSON file:
```json
{
    "book_title": {
        "character_name": [
            {
                "Multiple Choice Question": ...
            }
        ]
    },
    ...
}
``` 

Below are example command lines for motivation recognition:

```bash
cd code
python evaluation_motivation.py --persona_file ../exp/incremental/gpt-4-0125/result.json --type incremental --num_attempts 3
```

## Contact
If you have any questions, please contact [Xinfeng Yuan](xfyuan23@m.fudan.edu.cn), [Siyu Yuan](syyuan21@m.fudan.edu.cn), [Yuhan Cui](yhcui20@fudan.edu.cn).

## Citation
If you find our paper or resources useful, please kindly cite our paper.

```
@article{yuan2024evaluating,
      title={Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works}, 
      author={Xinfeng Yuan and Siyu Yuan and Yuhan Cui and Tianhe Lin and Xintao Wang and Rui Xu and Jiangjie Chen and Deqing Yang},
      year={2024},
      journal={arXiv preprint arXiv:2404.12726},
}
```