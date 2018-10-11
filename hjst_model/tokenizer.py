import re
import os
import MeCab
import shutil
try:
    import sentencepiece as spm
except:
    print('WARNING: you cannot use sentencepiece model')
    

class Morph(object):
    def __init__(self, tagging_mode = ' '):
        self.tagger = MeCab.Tagger(tagging_mode)
        self.tagger.parse('')

    def iter_surface(self, text):
        morph_list = self.tagger.parseToNode(text)
        while morph_list:
            yield morph_list.surface
            morph_list = morph_list.next

    def surfaces(self, text):
        return list(self.iter_surface(text))
    
def load_spm(savepath):
    model_path = os.path.join(savepath, 'model.model')
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
    
def get_spm(dataset, savepath, vocab_size=16000):
    model_path = os.path.join(savepath, 'model.model')
    vocab_path = os.path.join(savepath, 'model.vocab')
    corpus_path = os.path.join(savepath, 'corpus.txt')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        dataset.set_iterator_mode("Sentence", tag=False, sentence=True, tokenizer=lambda x: x)
        with open(corpus_path, 'w') as f:
            f.write('\n'.join(dataset))
        spm.SentencePieceTrainer.Train('--input={} --model_prefix=model --vocab_size={}'.format(corpus_path, vocab_size))
        shutil.move('./model.model', model_path)
        shutil.move('./model.vocab', vocab_path)
    return load_spm(savepath)
