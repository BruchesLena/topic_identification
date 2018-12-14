from sentiment_demo.sent_predictor_eng import Predictor

predictor = Predictor()
while True:
    text = input('text:\n')
    predictions = predictor.predict(text)
    for p in predictions:
        print(p[0], str(p[1]))