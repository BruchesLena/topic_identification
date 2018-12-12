from topics_demo.predictor_rus import Predictor

predictor = Predictor()
while True:
    text = input('text:\n')
    topics = predictor.predict(text)
    for topic in topics:
        print(topic[0], str(topic[1]))