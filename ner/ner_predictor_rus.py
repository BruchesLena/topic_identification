# -*- coding: utf-8 -*-
from nltk import wordpunct_tokenize
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Bidirectional, Lambda
import numpy as np
from keras_contrib.layers import CRF

class NerParser:

    def __init__(self):
        sess = tf.Session()
        K.set_session(sess)
        self.elmo_model = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                                trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        self.max_len = 100
        self.batch_size = 24
        self.load_model()
        self.tags = ['B-Project', 'O', 'B-Location', 'I-Project', 'B-Org', 'I-Org', 'I-Location', 'B-Person', 'I-Person']

    def load_model(self):
        def ElmoEmbedding(x):
            return self.elmo_model(inputs={
                "tokens": tf.squeeze(tf.cast(x, tf.string)),
                "sequence_len": tf.constant(24 * [100])
            },
                signature="tokens",
                as_dict=True)["elmo"]

        input_text = Input(shape=(100,), dtype="string")
        embedding = Lambda(ElmoEmbedding, output_shape=(100, 1024))(input_text)
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        crf = CRF(9, sparse_target=True)  # CRF layer
        out = crf(x)  # output

        self.model = Model(input_text, out)
        self.model.load_weights("D:\\NER\\weights_ner_elmo_crf.h5")
        print('model loaded')

    def vectorize_elmo(self, tokens):
        X = [[w for w in tokens]]
        new_X = []
        for seq in X:
            new_seq = []
            for i in range(self.max_len):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("__PAD__")
            new_X.append(new_seq)
        for i in range(24):
            new_seq = []
            for y in range(self.max_len):
                new_seq.append("__PAD__")
            new_X.append(new_seq)
        X = new_X
        return X

    def parse(self, text):
        tokens = wordpunct_tokenize(text)
        X_elmo = self.vectorize_elmo(tokens)
        q=0
        vector_elmo = X_elmo[q:q + self.batch_size]
        vector_elmo = np.array(vector_elmo)
        p = self.model.predict([vector_elmo])[0]
        p = np.argmax(p, axis=-1)
        parsed = []
        for token, pred in zip(tokens, p):
            parsed.append((token, self.tags[int(pred)]))
        return self.extract_spans(parsed)

    def extract_spans(self, parsed_tokens):
        chunk = ""
        ner_type = ""
        spans = []
        for i in range(len(parsed_tokens)):
            predicted_ner_type = parsed_tokens[i][1]
            token = parsed_tokens[i][0]
            if predicted_ner_type == 'O':
                if len(chunk) > 0:
                    spans.append((chunk[:-1], ner_type))
                    chunk = ""
                    ner_type = ""
                continue
            if predicted_ner_type.startswith('B'):
                if len(chunk) > 0:
                    spans.append((chunk[:-1], ner_type))
                    chunk = token + ' '
                    ner_type = predicted_ner_type[2:]
                else:
                    chunk = token + ' '
                    ner_type = predicted_ner_type[2:]
            if predicted_ner_type.startswith('I'):
                chunk += token + ' '
                if ner_type == "":
                    ner_type = predicted_ner_type[2:]
        if chunk != "":
            spans.append((chunk[:-1], ner_type))
        return spans
