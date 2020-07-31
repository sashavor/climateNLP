import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import sys
import json


def split_text_into_sentences(fpath):
    """
        input: file path to text file
        purpose: split text into sentences (list of strings)
        returns: sentences
    """
    
    # load in file
    with open(fpath, 'r') as fi:
        whole_text = fi.read()
    
    # init tokenizer
    tokenizer = PunktSentenceTokenizer()
    
    # remove newline characters from text
    whole_text = whole_text.replace("\n", "")
    
    # split text into sentences and remove empty sentences
    sentences = [sent for sent in tokenizer.tokenize(whole_text) if sent is not ""]
    
    return sentences



def text_to_tsv(input_dir, output_dir, question_file):
    
    # get list of files in input directory
    files = os.listdir(input_dir)
    
    out_fpaths = []
    
    # iterate through input files
    for fname in files:
        if fname[0] != ".":
            # split text into sentences
            fpath = input_dir.strip("/") + "/" + fname
            sentences = split_text_into_sentences(fpath)

            # generate tsv name
            out_fname = fname.split(".")[0] + ".tsv"
            out_fpath = output_dir.strip("/") + "/" + out_fname
            
            # get questions dict:
            question_dict = load_questions(question_file)
            
            # make dataframe with relevant information (sentences, questions, report name)
            curr_df = make_df(sentences, question_dict, fname.split(".")[0])
                        

            # save current dataframe as tsv
            curr_df.to_csv(out_fpath, sep="\t", index=False)
            
            out_fpaths.append(out_fpath)
            
    return out_fpaths


            
            

def load_questions(q_file):
    """
        input: json file containing questions, labeled by question number
            ex: {1: "QUESTION 1 TEXT",
                 2: "QUESTION 2 TEXT"}
        purpose: 
            - load in questions from the above format
        output: dictionary of questions in same format as above
    """
    with open(q_file) as json_file: 
        question_dict = json.load(json_file) 
  
    return question_dict



def make_df(sentence_list, question_dict, report_name):
    columns = ["TEXT_SNIPPET", "REPORT_NAME", "Q_NUMBER", "Q_TEXT"]
        
    q_num_list = sorted(list(question_dict.keys()))
    base_question_list = []

    for key in q_num_list:
        base_question_list.append(question_dict[key])    
    
    questions_repeated = []
    q_nums_repeated = []
    sentences_repeated = []
    
    # generate other necessary lists (report name list)
    for sentence in sentence_list:
        # repeat sentence the number of times of the base question list
        repeated_sentence = len(base_question_list) * [sentence]
        # add repeated sentence to sentences repeated list
        sentences_repeated = sentences_repeated + repeated_sentence
        # do the same for the questions and question numbers
        questions_repeated = questions_repeated + base_question_list
        q_nums_repeated = q_nums_repeated + q_num_list
        
    # generate a list of report names of the necessary length
    report_names = len(sentences_repeated) * [report_name]
    
    
    df = pd.DataFrame(list(zip(sentences_repeated, report_names, q_nums_repeated, questions_repeated)), 
                                columns=columns)
    return df
    


def main(argv):
    input_dir = argv[0]
    output_dir = argv[1]
    question_file = argv[2]
    print("Input directory is %s"%input_dir)
    print("Output directory is %s" % output_dir)
    print("Question file is %s" % question_file)
    
    text_to_tsv(input_dir, output_dir, question_file)

    
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
    
    
    