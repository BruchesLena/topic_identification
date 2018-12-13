from flair.data import Sentence
from flair.models import SequenceTagger

class NerPredictorEng:

    def __init__(self):
        pass

    def predict(self, text):
        sentence = Sentence(text)
        tagger = SequenceTagger.load('ner')
        tagger.predict(sentence)
        predictions = []
        for entity in sentence.get_spans('ner'):
            ids = []
            for token in entity.tokens:
                ids.append(token.text)
            predictions.append((' '.join(ids), entity.tag))
        return predictions

