import logging
import pkg_resources
import regex as re
import sentencepiece as spm

from .sockeye_handler import SockeyeHandler
from .text_processor import JoshuaPreprocessor, Detokenizer

BPE_MODEL = 'bpe.model'

class SPMEncode(JoshuaPreprocessor):
    def __init__(self, model=None):
        if model is None:
            model = BPE_MODEL
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model)

    def run(self, string):
        result = self.sp.EncodeAsPieces(string)
        return ' '.join(result)

class SPMDecode():
    def run(self, tokens):
        #this is the same detok method recommended in the SPM paper:
        #https://arxiv.org/pdf/1808.06226.pdf
        #i.e. split the line to eat the fake spaces, join tokens on 
        #spaces, and replace the space placeholder with a normal space
        result = ''.join(tokens.split()).replace("▁", " ")
        return result

class De2EnHandler(SockeyeHandler):
    """
    Consumes German text of arbitrary length and returns its translation in English.
    """

    def initialize(self, context):
        super().initialize(context)
        self.preprocessor = SPMEncode() 
        self.postprocessor = SPMDecode()


_service = De2EnHandler()


def handle(data, context):
    return _service.handle(data, context)
