from ner.ner_predictor_eng import NerPredictorEng

predictor = NerPredictorEng()
while True:
    text = input('text:\n')
    ners = predictor.predict(text)
    for ner in ners:
        print(ner)