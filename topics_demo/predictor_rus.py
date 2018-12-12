import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Bidirectional, Lambda


class Predictor:

    def __init__(self):
        sess = tf.Session()
        K.set_session(sess)
        self.elmo_model = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                                trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print("Done")
        self.tags = ['Оружие', 'Из жизни', 'Легпром', 'Культура', 'Бывший СССР', 'Сочи', 'Россия', 'Ценности', 'ЧМ-2014', 'МедНовости', 'Спорт', 'Экономика', 'Дом', 'Путешествия', 'Интернет и СМИ', 'Бизнес', 'Крым', 'Наука и техника', 'Силовые структуры', 'Культпросвет ', 'Библиотека', 'Мир']
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.n_tags = len(self.tags)
        self.max_len = 30
        self.batch_size = 25
        self.load_model()

    def ElmoEmbedding(self, x):
        return self.elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(self.batch_size * [self.max_len])
        },
            signature="tokens",
            as_dict=True)["elmo"]

    def load_model(self):
        input_text = Input(shape=(self.max_len,), dtype="string")
        embedding = Lambda(self.ElmoEmbedding, output_shape=(self.max_len, 1024))(input_text)
        lstm = Bidirectional(LSTM(512))(embedding)
        l_dense = Dense(100, activation='relu')(lstm)
        out = Dense(self.n_tags, activation='softmax')(l_dense)
        self.model = Model([input_text], out)
        self.model.load_weights('D:\\topics\\experiments\\weights2.h5')

    def vectorize_text(self, text):
        X_vector = text.split(' ')
        X = []
        new_seq = []
        for i in range(self.max_len):
            try:
                new_seq.append(X_vector[i])
            except:
                new_seq.append("__PAD__")
        X.append(new_seq)
        for i in range(24):
            new_seq = []
            for y in range(self.max_len):
                new_seq.append("__PAD__")
            X.append(new_seq)
        return np.array(X)

    def predict(self, text):
        vector = self.vectorize_text(text)
        predictions = self.model.predict(vector)[0]
        sorted_p = list(np.argsort(predictions))
        sorted_predictions = []
        for ind in sorted_p[::-1]:
            if predictions[ind] > 0.01:
                sorted_predictions.append((self.tags[ind],predictions[ind]))
        return sorted_predictions



