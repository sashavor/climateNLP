from allennlp.modules import Seq2VecEncoder
import torch
from overrides import overrides

@Seq2VecEncoder.register("bert-sentence-pooler")
class BertSentencePooler(Seq2VecEncoder):
    """

    """

    def __init__(self,
                bert_dim: int = 512
                ):
        self.bert_dim = bert_dim
        super().__init__()

    def forward(self,
                embs: torch.tensor,
                mask: torch.tensor = None
                ) -> torch.tensor:
        return embs[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        return self.bert_dim

###
