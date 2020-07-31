from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoriacalAccuracy

@Model.register('linear-classifier')
class LinearClassifier(Model):
    def __init__(self,
                vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                seq2vec_encoder: Seq2VecEncoder,
                dropout: float = 0.,
                label_namespace: str = 'labels',
                initializer: InitializerApplicator = InitializerApplicator()
                ) -> None:

