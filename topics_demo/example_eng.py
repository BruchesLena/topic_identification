from topics_demo.predictor_eng import PredictorEnglish

predictor = PredictorEnglish()
while True:
    text = input('text:\n')
    topics = predictor.predict(text)
    for topic in topics:
        print(topic[0], str(topic[1]))