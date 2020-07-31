from typing import Dict, Iterable, List, Optional, Callable, TypeVar
import logging

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, MetadataField
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer

from allennlp.common import Params

from overrides import overrides
import pandas as pd

T = TypeVar("T", bound="FromParams")


logger = logging.getLogger(__name__)


@DatasetReader.register('ccqa')
class CCQADatasetReader(DatasetReader):
    """
        Reads tsv dataset
    """

    def __init__(self,
                tokenizer: Optional[Tokenizer] = None,
                token_indexers: Dict[str, TokenIndexer] = None,
                combine_input_fields: Optional[bool] = True,
                question_col: str = 'Q_TEXT',
                text_col: str = 'TEXT_SNIPPET',
                label_col: str = None,
                max_len: int = 512,
                **kwargs
                ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or PretrainedTransformerTokenizer()

        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)
        self.question_col = question_col
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len

    @overrides
    def _read(self, file_path: str):
        # read in dataset as dataframe
        logger.info("Reading CCQA instances from tsv dataset at: %s ..." % (file_path))
        df = pd.read_csv(file_path, sep="\t")
        print("Loaded in df, iterating through rows...")
        # iterate through dataframe
        # NOTE: there's for sure a more efficient way to do this
        for index,row in df.iterrows():
            # create text instances
            if self.label_col:
                yield self.text_to_instance(row[self.question_col], row[self.text_col], row,  row[self.label_col])
            else:
                yield self.text_to_instance(row[self.question_col], row[self.text_col], row, label=None)
            #except Exception as e:
             #   print("Invalid data:")
              #  print(index)
               # print(row)


    @overrides
    def text_to_instance(self,
                        question: str,
                        text: str,
                        row,
                        label: str = None
                        ) -> Instance:
        #print("in tti")
        # init fields dict
        fields: Dict[str, Field] = {}
        #print("fields generated")
        # tokenize question and text
        question = self._tokenizer.tokenize(question)
        text = self._tokenizer.tokenize(text)
        #print("question and text tokenized")
        #print("combining sentences")
        # if want to combine the two sentences,
        if self._combine_input_fields:
            # combine them and then store them in one dict entry
            tokens = self._tokenizer.add_special_tokens(question, text)
            if len(tokens) > self.max_len:
                tokens = self.shorten(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            # else, add special tokens to them separately
            question_tokens = self._tokenizer.add_special_tokens(question)
            text_tokens = self._tokenizer.add_special_tokens(text)

            fields["question"] = TextField(question_tokens, self._token_indexers)
            fields["text"] = TextField(text_tokens, self._token_indexers)

        #if len(fields["tokens"]) > self.max_len:
        #    final_sep = fields["tokens"][-1:]
        #    new_tokens = fields["tokens"][:self.max_len-1] + final_sep
        #    fields["tokens"] = new_tokens



        #print("adding label to field (if labeled)")
        # if data is labeled, add label to field
        if label:
            fields["label"] = LabelField(label)
        #print("returning ...")
        return Instance(fields)
    

    def shorten(self, token_field) -> List[Token]:
        final_sep = token_field[-1:]
        shorter_tokens = token_field[:self.max_len-1] + final_sep
        return shorter_tokens
    
    
    



@DatasetReader.register('ccqa_contextual')
class CCQAContextualDatasetReader(DatasetReader):
    """
        Reads tsv dataset
    """

    def __init__(self,
                tokenizer: Optional[Tokenizer] = None,
                token_indexers: Dict[str, TokenIndexer] = None,
                combine_input_fields: Optional[bool] = True,
                question_col: str = 'Q_TEXT',
                text_col: str = 'TEXT_SNIPPET',
                label_col: str = 'IS_ANS_TO_Q',
                max_len: int = 512,
                **kwargs
                ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or PretrainedTransformerTokenizer()

        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)
        self.question_col = question_col
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len

    @overrides
    def _read(self, file_path: str):
        # read in dataset as dataframe
        logger.info("Reading CCQA instances from tsv dataset at: %s ..." % (file_path))
        df = pd.read_csv(file_path, sep="\t")
        print("Loaded in df, iterating through rows...")
        # iterate through dataframe
        # NOTE: there's for sure a more efficient way to do this
        for index,row in df.iterrows():
            report_name = row["REPORT_NAME"]
            # if isn't the beginning of a document
            if index - 14 > 0 and df.iloc[index-14]["REPORT_NAME"] == report_name:
                before_context = df.iloc[index-14][self.text_col]
            else:
                before_context = ""
                
            if index + 14 < df.shape[0] - 1 and df.iloc[index+14]["REPORT_NAME"] == report_name:
                after_context = df.iloc[index+14][self.text_col]
            else:
                after_context = ""
            # if data is ok, create text instances
            yield self.text_to_instance(row[self.question_col], 
                                        before_context, 
                                        row[self.text_col], 
                                        after_context, row,  
                                        row[self.label_col])

    @overrides
    def text_to_instance(self,
                        question: str,
                        before_context: str,
                        text: str,
                        after_context: str,
                        row,
                        label: str = None
                        ) -> Instance:
        # init fields dict
        fields: Dict[str, Field] = {}
        # tokenize question and text
        question = self._tokenizer.tokenize(question)
        before_context = self._tokenizer.tokenize(before_context)
        text = self._tokenizer.tokenize(text)
        after_context = self._tokenizer.tokenize(after_context)
        # combine them and then store them in one dict entry
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]
        
        import dataclasses
        
        # join question, text, and contexts, as well as add special tokens
        tokens = self._tokenizer.sequence_pair_start_tokens + \
                        with_new_type_id(question, self._tokenizer.sequence_pair_first_token_type_id) + \
                        self._tokenizer.sequence_pair_mid_tokens + \
                        with_new_type_id(before_context, self._tokenizer.sequence_pair_second_token_type_id) + \
                        self._tokenizer.sequence_pair_mid_tokens + \
                        with_new_type_id(text, self._tokenizer.sequence_pair_second_token_type_id) + \
                        self._tokenizer.sequence_pair_mid_tokens + \
                        with_new_type_id(after_context, self._tokenizer.sequence_pair_second_token_type_id) + \
                        self._tokenizer.sequence_pair_end_tokens
        
        if len(tokens) > self.max_len:
            tokens = self.shorten(tokens)
        fields["tokens"] = TextField(tokens, self._token_indexers)
#         else:
#             # else, add special tokens to them separately
#             question_tokens = self._tokenizer.add_special_tokens(question)
#             text_tokens = self._tokenizer.add_special_tokens(text)

#             fields["question"] = TextField(question_tokens, self._token_indexers)
#             fields["text"] = TextField(text_tokens, self._token_indexers)

        #if len(fields["tokens"]) > self.max_len:
        #    final_sep = fields["tokens"][-1:]
        #    new_tokens = fields["tokens"][:self.max_len-1] + final_sep
        #    fields["tokens"] = new_tokens


        #print("creating metadata dict")
        # create metadata dict and add it to the field
        metadata = {
            "company_name": row["COMPANY_NAME"],
            "sector": row["SECTOR"],
            "ticker": row["TICKER"],
            "country": row["COUNTRY"],
            "big_four": row["BIG_FOUR"],
            "report_name": row["REPORT_NAME"],
            "report_link": row["REPORT_LINK"],
            "q_number": row["Q_NUMBER"],
            "question": row["Q_TEXT"],
            "text": row["TEXT_SNIPPET"],
            "label": row["IS_ANS_TO_Q"]
        }
        #print("adding metadata field")
        #fields["metadata"] = MetadataField(metadata)
        #print("adding label to field (if labeled)")
        # if data is labeled, add label to field
        if label == "Y":
            label = "1"
        if label == "N":
            label = "0"
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)
    

    def shorten(self, token_field) -> List[Token]:
        final_sep = token_field[-1:]
        shorter_tokens = token_field[:self.max_len-1] + final_sep
        return shorter_tokens
    