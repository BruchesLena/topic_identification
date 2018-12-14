from ner.ner_predictor_rus import NerParser

ner = NerParser()
while True:
    text = input('text:\n')
    parsed = ner.parse(text)
    for p in parsed:
        print(p)