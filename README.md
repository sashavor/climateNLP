# climateNLP
Repository for summer 2020 internship on analyzing climate-related financial disclosures

## Description of the script
1. Reads in the arguments provided (description of which arguments are necessary is in the 'How to Run' section below)
2. Goes file by file and reads in all of the txt data in the input file. It splits the text into sentences, and creates a tsv file corresponding to each text file that contains all combinations of each sentence and each question. 
3. Goes tsv by tsv and runs predictions on each tsv, saving a version of the input tsv that also contains, for each question-sentence combination:
    - The model's prediction, Y or N (whether the sentence answers the given question)
    - The probability that the answer was Y
    - The probability that the answer was N

## How to Run:
From this directory, run the command (with the file paths):

<pre><code>python run.py INPUT_FOLDER_PATH OUTPUT_FOLDER_PATH QUESTION_JSON_FILE MODEL_PATH WEIGHTS_FILE</code></pre>


An Example is below:

<pre><code>python run.py Input_Data_Folder Output_Folder questions.json model.tar.gz best_weights.th</code></pre>

### Notes: 

- run.py
    - Included in this repository

- Input_Data_Folder
    - Some sample data can be found in the [Test-data](https://github.com/sashavor/climateNLP/tree/master/Test-data) folder
    - For training on your own data, this needs to be the path to a directory containing only the plaintext files (.txt) that contain the documents that you want to predict on
  
- Output Folder
    - Needs to be the path to a folder that already exists, and must be supplied
        - It does not matter if this folder is empty, but anything with the same file names as the outputs (e.g. outputs from a previous run of this script) will be overwritten without warning
  
- questions.json
    - A basic questions file (containing all the questions the model was trained on) is included in this repository. If you would like to run on a subset of the questions, input a path to a json file that contains only the subset of the questions that you want to run on. 
    - It is possible to run the model on questions outside of the set of questions here, however the model was not trained on those questions, so there is absolutely no guarantee on how well the model will do on those questions.

- model.tar.gz
    - The model that we trained on several years of sustainability reports (2016-2018) can be found [here](https://drive.google.com/file/d/1nRlMtx4X2LeIgmOHlzco9-ZcfwtKVwbH/view?usp=sharing).
    - It is possible to train your own model, but it must be saved in a tar.gz format, or another model format that [AllenNLP](https://github.com/allenai/allennlp) can automatically use to run its 'predict' command
 
- best_weights.th
    - The weights from our model can be found [here](https://drive.google.com/file/d/1xHVi70SOdIWsG0lWBtwOlWpK3N4XHnbk/view?usp=sharing)
    - It is possible to train your own model, but it must be saved in a .th format, or another weight format that [AllenNLP](https://github.com/allenai/allennlp) can automatically use to run its 'predict' command




