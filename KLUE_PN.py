import pandas as pd
from konlpy.tag import Mecab
import modeling2 as modeling
# import modeling_electra as modeling

import tensorflow as tf
import numpy as np

from utils import Fully_Connected
import optimization
import tokenization
from transformers import AutoTokenizer
from HTML_Utils import *
import os
import json
from evaluate2 import f1_score
from evaluate2 import exact_match_score

import Chuncker
import Ranking_ids

from HTML_Processor import process_document

import pos_augmented_tokenizer

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    Examples
    ---------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


class KoNET:
    def __init__(self, firstTraining, use_attention_supervision=True):
        self.use_attention_supervision = use_attention_supervision

        self.input_ids_arr = np.load('pn_data/input_ids.npy')
        self.token_type_ids_arr = np.load('pn_data/token_type_ids.npy')
        self.label_arr = np.load('pn_data/label.npy')
        self.attention_supervision_arr = np.load('pn_data/pos_attention.npy')

        self.ix = 0
        self.r_ix = np.array(range(self.input_ids_arr.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

        self.chuncker = Chuncker.Chuncker()
        self.first_training = firstTraining

        self.save_path = './pn_data/model.ckpt'
        # self.bert_path = '/home/ai/pycharm_project/koelectra/koelectra_base_v3'
        self.bert_path = '/home/ai/바탕화면/attention_super/nli_data/model/klue_bert/klue_bert.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label_tensor = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.pos_attention_matrix = tf.placeholder(shape=[None, None, None], dtype=tf.float32)

    def create_model(self, input_ids, input_segments, is_training=True, reuse=False):
        pos_attention_matrix = self.pos_attention_matrix
        if self.use_attention_supervision is False:
            pos_attention_matrix = None

        # bert_config = modeling.BertConfig.from_json_file('elec_config.json') # 32200
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')  # 32200
        #bert_config.vocab_size = 35000

        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))
        model = modeling.BertModel(
            # bert_config=bert_config,
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_segments,
            scope='bert',
            # scope='electra,'
            # reuse=reuse,
            additional_attention_matrix=pos_attention_matrix
        )

        bert_variables = tf.global_variables()

        return model, bert_variables, model.get_sequence_output()

    def get_nli_loss(self, logit):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=self.label_tensor)
        return loss

    def get_nli_probs(self, model_output, scope, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("MRC_block_" + scope):
            model_output = Fully_Connected(model_output, output=512, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=256, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            nli_probs = Fully_Connected(model_output, output=3, name='hidden', activation=None)

        return nli_probs

    def Training(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=True)

            nli_probs = self.get_nli_probs(model.get_pooled_output(), scope='text_layer', is_training=True)
            loss = self.get_nli_loss(nli_probs)
            loss = tf.reduce_mean(loss)
            total_loss = loss

            alpha = 150.0
            attention_supervision_loss = model.attention_entropy_loss

            # decay alpha
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = alpha
            decay_alpha = tf.train.exponential_decay(starter_learning_rate, global_step, 8, 0.94, staircase=True)

            if attention_supervision_loss is not None:
                total_loss = loss + attention_supervision_loss * decay_alpha

            learning_rate = 3e-5
            optimizer = optimization.create_optimizer(loss=total_loss, init_lr=learning_rate,
                                                      num_train_steps=training_epoch,
                                                      num_warmup_steps=int(training_epoch / 10),
                                                      use_tpu=False, global_step=global_step)
            sess.run(tf.initialize_all_variables())

            if self.first_training is True:
                saver = tf.train.Saver(bert_variables)
                saver.restore(sess, self.bert_path)
                print('elec restored') # everyday failed,,,

            batch_size = 64
            max_length = 64

            for i in range(training_epoch):
                input_ids = np.zeros(shape=[batch_size, max_length], dtype=np.int32)
                token_type_ids = np.zeros(shape=[batch_size, max_length], dtype=np.int32)
                attention_suervision_vector = np.zeros(shape=[batch_size, max_length, max_length], dtype=np.int32)
                labels = np.zeros(shape=[batch_size, 3], dtype=np.float32)

                for b in range(batch_size):
                    ix = self.r_ix[self.ix]
                    input_ids[b] = self.input_ids_arr[ix]
                    token_type_ids[b] = self.token_type_ids_arr[ix]
                    attention_suervision_vector[b] = self.attention_supervision_arr[ix]
                    labels[b] = self.label_arr[ix]
                    self.ix += 1

                if self.ix + batch_size >= self.input_ids_arr.shape[0]:
                    self.ix = 0

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: token_type_ids,
                             self.pos_attention_matrix: attention_suervision_vector,
                             self.label_tensor: labels
                             }

                loss_, loss2_, _, alpha_value = sess.run([total_loss, loss, optimizer, decay_alpha], feed_dict=feed_dict)

                if attention_supervision_loss is not None:
                    attention_supervision_loss_value = sess.run(attention_supervision_loss, feed_dict=feed_dict)
                    print(i, '/', training_epoch, loss_, loss2_, 'attention_loss:', attention_supervision_loss_value, alpha_value)

                else:
                    print(i, '/', training_epoch, loss_)

                # epoch_num = int(training_epoch / 20)
                # if i % epoch_num == 0 :
                #     saver = tf.train.Saver()
                #     saver.save(sess, self.save_path + '/' + str(int(i / epoch_num)) + '_nli.ckpt')

            saver = tf.train.Saver()
            saver.save(sess, self.save_path)

    def eval_nli_with_file(self):
        # save_path = self.save_path + '/' + str(epoch) + '_nli.ckpt'
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        cor = 0
        epo = 0

        all_input_ids = np.load('pn_data/input_ids_dev.npy')
        all_token_type_ids = np.load('pn_data/token_type_ids_dev.npy')
        all_label_ids = np.load('pn_data/label_dev.npy')
        pos_attention_label = np.load('pn_data/pos_attention_dev.npy')

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=False)
            pooled_output = model.get_pooled_output()

            nli_probs = self.get_nli_probs(pooled_output, scope='text_layer', is_training=False)
            nli_prediction = tf.argmax(nli_probs, axis=1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)
            cnt = 0

            max_length = 64

            for i in range(len(all_label_ids)):
                total_num = 1

                input_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
                token_type_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
                label_idx = np.zeros(shape=[total_num], dtype=np.int32)

                for j in range(total_num):
                    input_ids[j] = all_input_ids[cnt]
                    token_type_ids[j] = all_token_type_ids[cnt]
                    label_idx[j] = all_label_ids[cnt]
                    cnt += 1

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: token_type_ids,
                             self.pos_attention_matrix: pos_attention_label}

                prediction_idx = sess.run(nli_prediction, feed_dict=feed_dict)

                for j in range(total_num):
                    my_str = ''
                    for inp in input_ids:
                        my_str += tokenizer.decode(inp)
                    # print(my_str)
                    if prediction_idx[j] == label_idx[j]:
                        cor += 1
                    epo += 1

                print(cor, '/', epo, cor / epo)

    def eval_nli(self):
        data = json.load(open('./nli_data/klue-nli-v1.1_dev.json', 'r', encoding='utf-8'))
        label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        vocab = tokenization.load_vocab(vocab_file='koelec_vocab.txt')
        # tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        f_tokenizer = tokenization.FullTokenizer(vocab_file='koelec_vocab.txt')

        max_length = 64

        cor = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=False)
            pooled_output = model.get_pooled_output()

            nli_probs = self.get_nli_probs(pooled_output, scope='text_layer', is_training=False)
            nli_prediction = tf.argmax(nli_probs, axis=1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for data_dict in data:
                premise = data_dict['premise']
                hypothesis = data_dict['hypothesis']
                label_text = data_dict['gold_label']
                label_idx = label_dict[label_text]

                total_num = 1
                max_length = 64

                input_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
                token_type_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
                pos_attention_label = np.zeros(shape=[total_num, max_length, max_length], dtype=np.int32)

                tokens = ['[CLS]']
                seq_tokens = tokenizer.tokenize(premise)

                tokens.extend(seq_tokens)
                tokens.append('[SEP]')

                segments = [0] * len(tokens)

                seq_tokens = tokenizer.tokenize(hypothesis)
                tokens.extend(seq_tokens)
                segments.extend([1] * len(tokens))

                ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
                length = len(ids)
                if length > max_length:
                    length = max_length

                for j in range(length):
                    input_ids[0, j] = ids[j]
                    token_type_ids[0, j] = segments[j]
                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: token_type_ids,
                             self.pos_attention_matrix: pos_attention_label}

                prediction_idx = sess.run(nli_prediction, feed_dict=feed_dict)
                if prediction_idx[0] == label_idx:
                    cor += 1
                epo += 1

                print(cor, '/', epo, cor / epo)

    def eval_with_file(self):
        tagger = Mecab()
        chuncker = Chuncker.Chuncker()
        path_dir = '/data/korquad_data/korquad2_dev/'

        file_list = os.listdir(path_dir)
        file_list.sort()

        vocab = tokenization.load_vocab(vocab_file='koelec_vocab.txt')
        tokenizer = pos_augmented_tokenizer.TokenizerAugmentedPOS()
        #tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

        #vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        #tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='koelec_vocab.txt')

        max_length = 512

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=False)

            prob_start, prob_stop = self.get_qa_probs(sequence_output, scope='text_layer', is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for file_name in file_list:
                in_path = path_dir + file_name
                data = json.load(open(in_path, 'r', encoding='utf-8'))

                for article in data['data']:
                    doc = str(article['context'])
                    doc = doc.replace('\t', ' ')
                    doc = doc.replace('\a', ' ')

                    for qas in article['qas']:
                        answer = qas['answer']
                        answer_start = answer['answer_start']
                        answer_text = answer['text']
                        question = qas['question']

                        if len(answer_text) > 40:
                            continue

                        query_tokens = []
                        query_tokens.append('[CLS]')
                        q_tokens = tokenizer.tokenize(question.lower())
                        for tk in q_tokens:
                            query_tokens.append(tk)
                        query_tokens.append('[SEP]')

                        if True:
                            doc_ = doc[0: answer_start] + '[STA]' + answer_text + doc[answer_start + len(
                                answer_text): -1]
                            doc_ = str(doc_)

                            texts = []
                            new_doc = ''
                            seq = ''

                            splits = ['</ul>', '</p>', '</table>']

                            for i in range(len(doc)):
                                seq += doc_[i]

                                if i > 5:
                                    for spliter in splits:
                                        if doc_[i - len(spliter): i] == spliter:
                                            texts.append(seq)
                                            seq = ''
                            texts.append(seq)

                            for text in texts:
                                if text.find('<table>') != -1 and text.find('[STA]') != -1:
                                    text = text.replace('[STA]', '').replace('[END]', '')
                                    text = text.replace('<table>', '[STA] <table> [END]')

                                    table_case = True

                                new_doc += text + ' '

                            # doc_ = new_doc
                            doc_ = doc_.replace('<table>', ' <table> ')
                            doc_ = doc_.replace('</table>', ' </table> ')
                            doc_ = doc_.replace('<td>', ' , ')
                            doc_ = doc_.replace('</td>', ' ')
                            doc_ = doc_.replace('<th>', ' , ')
                            doc_ = doc_.replace('</th>', ' ')
                            doc_ = doc_.replace('<ul>', ' <ul> ')
                            doc_ = doc_.replace('</ul>', ' </ul> ')
                            doc_ = doc_.replace('<li>', ' , ')
                            doc_ = doc_.replace('</li>', ' ')
                            doc_ = doc_.replace('<p>', ' ')
                            doc_ = doc_.replace('</p>', ' ')

                            #
                            #####

                            doc_ = process_document(doc_)

                            paragraphs = doc_.split('[h2]')

                            sequences = []

                            for paragraph in paragraphs:
                                tokens = tokenizer.tokenize(paragraph)

                                try:
                                    title = paragraph.split('[/h2]')[0]
                                    paragraph = paragraph.split('[/h2]')[1]
                                except:
                                    title = ''

                                sub_paragraphs = paragraph.split('[h3]')

                                sequence = ''
                                total_length = 0
                                temp_queue = []

                                for sub_paragraph in sub_paragraphs:
                                    tokens = tokenizer.tokenize(sub_paragraph)
                                    if len(tokens) + len(query_tokens) > max_length:
                                        sub_sentences = sub_paragraph.replace('.', '.\n').split('\n')

                                        for sentence in sub_sentences:
                                            sentence += ' '
                                            sequence += sentence
                                            temp_queue.append(sentence)

                                            tokens = tokenizer.tokenize(sentence)

                                            if total_length + len(tokens) + len(query_tokens) + 30 >= max_length:
                                                sequences.append(title + ' ' + sequence)

                                                sequence = ' ' + sentence + ' '
                                                total_length = 0

                                                try:
                                                    for _ in range(2):
                                                        temp_sequence = temp_queue.pop(-1)
                                                        sequence = temp_sequence + ' ' + temp_sequence
                                                        total_length += len(tokenizer.tokenize(temp_sequence))
                                                except:
                                                    None

                                            total_length += len(tokens)
                                        sequences.append(title + ' ' + sequence)
                                    else:
                                        sequences.append(title + ' ' + sub_paragraph)

                            total_score = 0
                            for sequence in sequences:
                                score = chuncker.get_chunk_score(paragraph=sequence)
                                total_score += score
                            avg_score = total_score / len(sequences)

                            sequences_ = sequences
                            sequences = []

                            for sequence in sequences_:
                                score = chuncker.get_chunk_score(paragraph=sequence)

                                # if score < avg_score:
                                #    continue

                                if sequence.find('목차') != -1 and sequence.find('[list]') != -1:
                                    continue

                                if len(sequence) > 80:
                                    sequences.append(sequence)
                            ###################

                            sequences_ = []

                            chunk_scores = []
                            for sequence in sequences:
                                if sequence.find('[STA]') != -1:
                                    chunk_scores.append(1)
                                    sequences_.append(sequence)
                                else:
                                    chunk_scores.append(0)
                                #chunk_scores.append(chuncker.get_chunk_score(sequence))

                            sequences = sequences_

                            input_ids = np.zeros(shape=[len(sequences), max_length], dtype=np.int32)
                            input_mask = np.zeros(shape=[len(sequences), max_length], dtype=np.int32)
                            input_segments = np.zeros(shape=[len(sequences), max_length], dtype=np.int32)
                            pos_attention_matrix = np.zeros(shape=[len(sequences), max_length, max_length], dtype=np.int32)

                            have_to_pass = False

                            for b, sequence in enumerate(sequences):
                                if sequence.find('table') != -1 and sequence.find('[STA]') != -1:
                                    have_to_pass = True
                                    #break
                                sequence = sequence.replace('[STA]', '')

                                tokens = []
                                segments = []
                                seq_tokens, attention_matrix = tokenizer.make_attention_matrix(document=sequence, add=len(query_tokens))

                                tokens.extend(query_tokens)
                                segments.extend([0] * len(query_tokens))

                                tokens.extend(seq_tokens)
                                segments.extend([1] * len(seq_tokens))

                                ids = tokenization.convert_tokens_to_ids(tokens=tokens, vocab=vocab)
                                length = len(ids)
                                if length >= max_length:
                                    length = max_length

                                for j in range(length):
                                    input_ids[b, j] = ids[j]
                                    input_segments[b, j] = segments[j]


                            feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                                         self.input_segments: input_segments}

                            if input_ids.shape[0] > 0 and have_to_pass is False:
                                print(input_ids.shape, input_mask.shape)

                                probs_start, probs_stop = sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                                probs_start = np.array(probs_start, dtype=np.float32)
                                probs_stop = np.array(probs_stop, dtype=np.float32)

                                for j in range(input_ids.shape[0]):
                                    for k in range(1, input_ids.shape[1]):
                                        probs_start[j, k] = 0
                                        probs_stop[j, k] = 0

                                        if input_ids[j, k] == 3:
                                            break

                                self.chuncker.get_feautre(question)

                                prob_scores = []
                                c_scores = []

                                for j in range(input_ids.shape[0]):
                                    # paragraph ranking을 위한 score 산정기준
                                    score2 = -(probs_start[j, 0] + probs_stop[j, 0])
                                    prob_scores.append(score2)
                                    c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                                if True:
                                    for j in range(input_ids.shape[0]):
                                        probs_start[j, 0] = -999
                                        probs_stop[j, 0] = -999

                                    # CLS 선택 무효화

                                    prediction_start = probs_start.argsort(axis=1)[:, 512 - 5:512]
                                    prediction_stop = probs_stop.argsort(axis=1)[:, 512 - 5:-512]

                                    answers = []
                                    scores = []
                                    candi_scores = []

                                    for j in range(input_ids.shape[0]):
                                        answer_start_idxs = prediction_start[j]
                                        answer_end_idxs = prediction_stop[j]

                                        probs = []
                                        idxs = []

                                        for start_idx in answer_start_idxs:
                                            for end_idx in answer_end_idxs:
                                                if start_idx > end_idx:
                                                    continue
                                                if end_idx - start_idx > 100:
                                                    continue

                                                idxs.append([start_idx, end_idx])
                                                probs.append(probs_start[j, start_idx] + probs_stop[j, end_idx])

                                        start_prob = 0
                                        stop_prob = 0
                                        if len(probs) == 0:
                                            answer_start_idx = probs_start.argmax(axis=1)[j]
                                            answer_stop_idx = probs_stop.argmax(axis=1)[j]

                                        else:
                                            idx = np.array(probs).argmax()
                                            answer_start_idx = idxs[idx][0]
                                            answer_stop_idx = idxs[idx][1]

                                        #score = chunk_scores[j]
                                        start_prob = probs_start[j, answer_start_idx]
                                        stop_prob = probs_stop[j, answer_stop_idx]

                                        score = score * (2 + prob_scores[j]) * (start_prob + stop_prob)

                                        scores.append(score)
                                        candi_scores.append(score)

                                        if answer_start_idx > answer_stop_idx:
                                            answer_stop_idx = answer_start_idx + 15
                                        if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                            for k in range(answer_start_idx, input_ids.shape[1]):
                                                if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                                    answer_stop_idx = k
                                                    break

                                        answer = ''

                                        if answer_stop_idx + 1 >= input_ids.shape[1]:
                                            answer_stop_idx = input_ids.shape[1] - 2

                                        if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[table]':
                                            t_cnt = 0

                                            for k in range(answer_start_idx):
                                                tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                                if tok == '[table]':
                                                    t_cnt += 1

                                            answer = table_total_list[j][t_cnt]
                                        else:
                                            for k in range(answer_start_idx, answer_stop_idx + 1):
                                                tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                                if len(tok) > 0:
                                                    if tok[0] != '#':
                                                        answer += ' '
                                                answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace('##', '')

                                        answers.append(answer)

                                if len(answers) > 0:
                                    answer_candidates = []
                                    candidates_scores = []

                                    for _ in range(1):
                                        m_s = -99
                                        m_ix = 0

                                        for q in range(len(scores)):
                                            if m_s < scores[q]:
                                                m_s = scores[q]
                                                m_ix = q

                                        answer_candidates.append(answer_re_touch(answers[m_ix]))
                                        candidates_scores.append(candi_scores[m_ix])
                                        print('score:', scores[m_ix])
                                        scores[m_ix] = -999

                                    a1 = []
                                    a2 = []

                                    for a_c in answer_candidates:
                                        a1.append(exact_match_score(prediction=a_c, ground_truth=answer_text))
                                        a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                                    em_total += max(a1)
                                    f1_total += max(a2)
                                    epo += 1
                                    """
                                    answer_idx = np.array(candidates_scores, dtype=np.float32).argmax()
                                    a1 = exact_match_score(prediction=answer_candidates[answer_idx], ground_truth=answer_text)
                                    a2 = f1_score(prediction=answer_candidates[answer_idx], ground_truth=answer_text)
                                    em_total += a1
                                    f1_total += a2
                                    epo += 1
                                    print('answer:', answer_candidates[answer_idx])
                                    """
                                    for j in range(input_ids.shape[0]):
                                        f1_ = f1_score(prediction=answer_re_touch(answers[j]), ground_truth=answer_text)

                                        text = answers[j]

                                        print('score:', scores[j], 'F1:', f1_, ' , ', text.replace('\n', ' '))
                                    print('question:', question)
                                    print('answer:', answer_text)
                                    print('EM:', em_total / epo)
                                    print('F1:', f1_total / epo)
                                    print(file_name)
                                    print('-----\n', epo)
                                    if epo >= 8000:
                                        print('complete')
                                        exit(-1)


    def eval_with_file2(self):
        chuncker = Chuncker.Chuncker()
        in_path = "KorQuAD_v1.0_dev.json"
        data = json.load(open(in_path, 'r'))

        vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        tokenizer = pos_augmented_tokenizer.TokenizerAugmentedPOS()
        #tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab_bigbird.txt')

        max_length = 512

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=False)

            prob_start, prob_stop = self.get_qa_probs(sequence_output, scope='text_layer', is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            data = json.load(open(in_path, 'r', encoding='utf-8'))
            for article in data['data']:
                for para in article['paragraphs']:
                    doc = str(para['context'])
                    doc = doc.replace('\t', ' ')
                    doc = doc.replace('\a', ' ')

                    for qas in para['qas']:
                        for answer in qas['answers']:
                            answer_start = answer['answer_start']
                            answer_text = answer['text']
                            question = qas['question']
                            break

                        if len(answer_text) > 40:
                            continue

                        query_tokens = []
                        query_tokens.append('[CLS]')
                        q_tokens = tokenizer.tokenize(question.lower())
                        for tk in q_tokens:
                            query_tokens.append(tk)
                        query_tokens.append('[SEP]')

                        input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
                        input_segments = np.zeros(shape=[1, max_length], dtype=np.int32)
                        pos_attention_matrix = np.zeros(shape=[1, max_length, max_length], dtype=np.int32)

                        tokens = []
                        segments = []

                        #seq_tokens = tokenizer.tokenize(doc)
                        #pos_attention = None
                        #"""
                        try:
                            seq_tokens, pos_attention = tokenizer.make_attention_matrix(document=doc, add=len(query_tokens))
                        except:
                            continue
                        #"""
                        tokens.extend(query_tokens)
                        segments.extend([0] * len(query_tokens))

                        tokens.extend(seq_tokens)
                        segments.extend([1] * len(seq_tokens))

                        ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

                        length = len(segments)
                        if length > max_length:
                            length = max_length

                        for j in range(length):
                            input_ids[0, j] = ids[j]
                            input_segments[0, j] = segments[j]
                        pos_attention_matrix[0] = pos_attention

                        feed_dict = {self.input_ids: input_ids,
                                     self.input_segments: input_segments,
                                     self.pos_attention_matrix: pos_attention_matrix
                                     }

                        probs_start, probs_stop = sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                        probs_start = np.array(probs_start, dtype=np.float32)
                        probs_stop = np.array(probs_stop, dtype=np.float32)

                        for j in range(input_ids.shape[0]):
                            for k in range(1, input_ids.shape[1]):
                                probs_start[j, k] = 0
                                probs_stop[j, k] = 0

                                if input_ids[j, k] == 3:
                                    break

                        self.chuncker.get_feautre(question)

                        prob_scores = []
                        c_scores = []

                        for j in range(input_ids.shape[0]):
                            # paragraph ranking을 위한 score 산정기준
                            score2 = -(probs_start[j, 0] + probs_stop[j, 0])
                            prob_scores.append(score2)

                        if True:
                            for j in range(input_ids.shape[0]):
                                probs_start[j, 0] = -999
                                probs_stop[j, 0] = -999

                            # CLS 선택 무효화

                            prediction_start = probs_start.argmax(axis=1)
                            prediction_stop = probs_stop.argmax(axis=1)

                            answers = []
                            scores = []
                            candi_scores = []

                            for j in range(input_ids.shape[0]):
                                answer_start_idx = prediction_start[j]
                                answer_stop_idx = prediction_stop[j]

                                if answer_start_idx > answer_stop_idx:
                                    answer_stop_idx = answer_start_idx + 15
                                if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                    for k in range(answer_start_idx, input_ids.shape[1]):
                                        if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                            answer_stop_idx = k
                                            break

                                answer = ''

                                if answer_stop_idx + 1 >= input_ids.shape[1]:
                                    answer_stop_idx = input_ids.shape[1] - 2

                                for k in range(answer_start_idx, answer_stop_idx + 1):
                                    tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                    if len(tok) > 0:
                                        if tok[0] != '#':
                                            answer += ' '
                                    answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace('##', '')

                                answers.append(answer)

                        if len(answers) > 0:
                            answer_candidates = []
                            candidates_scores = []

                            for _ in range(1):
                                m_ix = 0
                                answer_candidates.append(answer_re_touch(answers[m_ix]))

                            a1 = []
                            a2 = []

                            for a_c in answer_candidates:
                                a1.append(exact_match_score(prediction=a_c, ground_truth=answer_text))
                                a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                            em_total += max(a1)
                            f1_total += max(a2)
                            epo += 1
                            """
                            answer_idx = np.array(candidates_scores, dtype=np.float32).argmax()
                            a1 = exact_match_score(prediction=answer_candidates[answer_idx], ground_truth=answer_text)
                            a2 = f1_score(prediction=answer_candidates[answer_idx], ground_truth=answer_text)
                            em_total += a1
                            f1_total += a2
                            epo += 1
                            print('answer:', answer_candidates[answer_idx])
                            """
                            print('question:', question)
                            print('answer:', answer_text)
                            print('EM:', em_total / epo)
                            print('F1:', f1_total / epo)
                            print('-----\n', epo)

    def eval_with_file3(self):
        df = pd.read_excel('finetune_data.xls')

        sentences = df['문장']
        questions = df['질문']
        answer_texts = df['정답']

        vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        tokenizer = pos_augmented_tokenizer.TokenizerAugmentedPOS()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab_bigbird.txt')

        max_length = 512

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=False)

            prob_start, prob_stop = self.get_qa_probs(sequence_output, scope='text_layer', is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for i in range(len(questions)):
                question = '철수가 ' + questions[i]

                answer_text = answer_texts[i]
                doc = sentences[i]

                query_tokens = []
                query_tokens.append('[CLS]')
                q_tokens = tokenizer.tokenize(question.lower())
                for tk in q_tokens:
                    query_tokens.append(tk)
                query_tokens.append('[SEP]')

                input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
                input_segments = np.zeros(shape=[1, max_length], dtype=np.int32)
                pos_attention_matrix = np.zeros(shape=[1, max_length, max_length], dtype=np.int32)

                tokens = []
                segments = []

                seq_tokens, pos_attention = tokenizer.make_attention_matrix(document=doc, add=len(query_tokens))

                tokens.extend(query_tokens)
                segments.extend([0] * len(query_tokens))

                tokens.extend(seq_tokens)
                segments.extend([1] * len(seq_tokens))

                ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

                length = len(segments)
                if length > max_length:
                    length = max_length

                for j in range(length):
                    input_ids[0, j] = ids[j]
                    input_segments[0, j] = segments[j]
                pos_attention_matrix[0] = pos_attention

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.pos_attention_matrix: pos_attention_matrix
                             }

                probs_start, probs_stop = sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                probs_start = np.array(probs_start, dtype=np.float32)
                probs_stop = np.array(probs_stop, dtype=np.float32)

                for j in range(input_ids.shape[0]):
                    for k in range(1, input_ids.shape[1]):
                        probs_start[j, k] = 0
                        probs_stop[j, k] = 0

                        if input_ids[j, k] == 3:
                            break

                for j in range(input_ids.shape[0]):
                    for k in range(1, input_ids.shape[1]):
                        if input_ids[j, k] == 0:
                            probs_start[j, k] = 0
                            probs_stop[j, k] = 0

                self.chuncker.get_feautre(question)

                prob_scores = []
                c_scores = []

                for j in range(input_ids.shape[0]):
                    # paragraph ranking을 위한 score 산정기준
                    score2 = -(probs_start[j, 0] + probs_stop[j, 0])
                    prob_scores.append(score2)

                if True:
                    for j in range(input_ids.shape[0]):
                        probs_start[j, 0] = -999
                        probs_stop[j, 0] = -999

                    # CLS 선택 무효화

                    prediction_start = probs_start.argsort(axis=1)[:, 512 - 10:512]
                    prediction_stop = probs_stop.argsort(axis=1)[:, 512 - 10:512]

                    answers = []
                    scores = []
                    candi_scores = []

                    for j in range(input_ids.shape[0]):
                        answer_start_idxs = prediction_start[j]
                        answer_end_idxs = prediction_stop[j]

                        probs = []
                        idxs = []

                        for start_idx in answer_start_idxs:
                            for end_idx in answer_end_idxs:
                                if start_idx > end_idx:
                                    continue
                                if end_idx - start_idx > 1:
                                    continue

                                idxs.append([start_idx, end_idx])
                                probs.append(probs_start[j, start_idx] + probs_stop[j, end_idx])

                        if len(probs) == 0:
                            print('@')
                            answer_start_idx = probs_start.argmax(axis=1)[j]
                            answer_stop_idx = probs_stop.argmax(axis=1)[j]
                        else:
                            idx = np.array(probs).argmax()
                            answer_start_idx = idxs[idx][0]
                            answer_stop_idx = idxs[idx][1]
                            print()
                        if answer_start_idx > answer_stop_idx:
                            answer_stop_idx = answer_start_idx
                        if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                            for k in range(answer_start_idx, input_ids.shape[1]):
                                if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                    answer_stop_idx = k
                                    break

                        answer = ''

                        if answer_stop_idx + 1 >= input_ids.shape[1]:
                            answer_stop_idx = input_ids.shape[1] - 2

                        for k in range(answer_start_idx, answer_stop_idx + 1):
                            tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                            if len(tok) > 0:
                                if tok[0] != '#':
                                    answer += ' '
                            answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace('##', '')

                        answers.append(answer)

                if len(answers) > 0:
                    answer_candidates = []
                    candidates_scores = []

                    for _ in range(1):
                        m_ix = 0
                        answer_candidates.append(answer_re_touch(answers[m_ix]))

                    a1 = []
                    a2 = []

                    for a_c in answer_candidates:
                        a1.append(exact_match_score(prediction=a_c, ground_truth=answer_text))
                        a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                    em_total += max(a1)
                    f1_total += max(a2)
                    epo += 1

                    print(doc)
                    print('question:', question)
                    print('answer:', answer_text, 'prediction:', answers[0])
                    print('EM:', em_total / epo)
                    print('-----\n', epo)
                    if epo >= 8000:
                        print('complete')
                        exit(-1)

    def eval_with_span(self):
        def clean_tokenize(query):
            bert_tokens = tokenizer.tokenize(query)
            #print(bert_tokens)
            tokens = []
            pre_text = ""
            for i in range(len(bert_tokens)):
                bert_token = bert_tokens[i].replace("##", "")
                if i + 1 < len(bert_tokens):
                    post_token = bert_tokens[i + 1].replace("##", "")
                else:
                    post_token = ""
                if bert_token == '[UNK]':
                    token = str(
                        re.match(f"{pre_text}(.*){post_token}(.*)",
                                 query).group(1))
                    tokens.append(token)
                    pre_text += token
                else:
                    tokens.append(bert_token)
                    pre_text += bert_token
            return tokens

        file = open('qtype.csv', 'r', encoding='utf-8')
        lines = file.read().split('\n')

        queries = []
        codes = []

        for line in lines:
            tk = line.split(',')
            try:
                codes.append(int(tk[1]))
                queries.append(tk[0])
            except:
                continue

        #print(len(queries), len(codes))
        #input()

        #name_tagger = Name_Tagging.Name_tagger()
        chuncker = Chuncker.Chuncker()

        path_dir = '/data/korquad_data/korquad2_dev'

        file_list = os.listdir(path_dir)
        file_list.sort()

        vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

        #vocab = tokenization.load_vocab(vocab_file='vocab.txt')
        #tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab_bigbird.txt')

        em_total = 0
        f1_total = 0
        epo = 0

        em_total1 = 0
        f1_total1 = 0
        epo1 = 0

        em_total2 = 0
        f1_total2 = 0
        epo2 = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments,
                                                                       is_training=False)
            prob_start, prob_stop = self.get_qa_probs(sequence_output, scope='text_layer', is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            num_tag = ['가장', '더', '제일', '많이', '적게']

            for file_name in file_list:
                print(file_name, 'processing evaluation')

                in_path = path_dir + '/' +file_name
                data = json.load(open(in_path, 'r', encoding='utf-8'))

                for article in data['data']:
                    doc = str(article['context'])

                    for qas in article['qas']:
                        error_code = -1

                        answer = qas['answer']
                        answer_start = answer['answer_start']
                        answer_text = answer['text']
                        question = qas['question']

                        chuncker.get_feautre(query=question)

                        if len(answer_text) > 40:
                            continue

                        query_tokens = []
                        query_tokens.append('[CLS]')
                        q_tokens = tokenizer.tokenize(question.lower())
                        for tk in q_tokens:
                            query_tokens.append(tk)
                        query_tokens.append('[SEP]')

                        ######
                        # 정답에 ans 토큰을 임베딩하기 위한 코드
                        ######

                        ans1 = ''
                        ans2 = ''
                        if doc[answer_start - 1] == ' ':
                            ans1 = '[answer]'
                        else:
                            ans1 = '[answer]'

                        if doc[answer_start + len(answer_text)] == ' ':
                            ans2 = '[/answer]'
                        else:
                            ans2 = '[/answer]'

                        doc_ = doc[0: answer_start] + ans1 + answer_text + ans2 + doc[
                                                                                  answer_start + len(answer_text): -1]
                        doc_ = str(doc_)
                        #
                        #####

                        paragraphs = doc_.split('<h2>')
                        sequences = []

                        checked = False

                        for paragraph in paragraphs:
                            try:
                                title = paragraph.split('[/h2]')[0]
                                paragraph = paragraph.split('[/h2]')[1]
                            except:
                                title = ''

                            sub_paragraphs = paragraph.split('<h3>')

                            for sub_paragraph in sub_paragraphs:
                                if checked is True:
                                    break

                                paragraph_, table_list = pre_process_document(paragraph, answer_setting=False,
                                                                              a_token1='',
                                                                              a_token2='')
                                paragraph = process_document(paragraph)

                                for table_text in table_list:
                                    if checked is True:
                                        break

                                    if table_text.find('[answer]') != -1:
                                        table_text = table_text.replace('[answer]', '')
                                        table_text = table_text.replace('[/answer]', '')

                                        table_text = table_text.replace('<th', '<td')
                                        table_text = table_text.replace('</th', '</td')

                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('[answer]<td>', '<td>[answer] ')
                                        table_text = table_text.replace('</td>[/answer]', ' [/answer]</td>')
                                        table_text = table_text.replace('</td>', '  </td>')
                                        table_text = table_text.replace('<td>', '<td> ')

                                        table_text = table_text.replace('<a>', '')
                                        table_text = table_text.replace('<b>', '')
                                        table_text = table_text.replace('</a>', '')
                                        table_text = table_text.replace('</b>', '')

                                        table_text, child_texts = overlap_table_process(table_text=table_text)
                                        table_text = head_process(table_text=table_text)

                                        table_holder.get_table_text(table_text=table_text)
                                        table_data = table_holder.table_data
                                        lengths = []

                                        for data in table_data:
                                            lengths.append(len(data))
                                        if len(lengths) <= 0:
                                            break

                                        length = max(lengths)

                                        rank_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        col_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        row_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)

                                        count_arr = np.zeros(shape=[200], dtype=np.int32)
                                        for data in table_data:
                                            count_arr[len(data)] += 1
                                        table_head = get_table_head(table_text=table_text, count_arr=count_arr)

                                        rankings = Ranking_ids.numberToRanking(table_data, table_head)

                                        for j in range(length):
                                            for i in range(len(table_data)):
                                                col_ids[i, j] = j
                                                row_ids[i, j] = i
                                                rank_ids[i, j] = rankings[i][j]

                                        idx = 0
                                        tokens_ = []
                                        clean_tokens_ = []
                                        rows_ = []
                                        cols_ = []
                                        ranks_ = []
                                        name_tags_ = []

                                        for i in range(len(table_data)):
                                            for j in range(len(table_data[i])):
                                                if table_data[i][j] is not None:
                                                    tokens = tokenizer.tokenize(table_data[i][j])
                                                    try:
                                                        clean_tokens = clean_tokenize(str(table_data[i][j]).strip())
                                                    except:
                                                        print('error:', table_data[i][j])
                                                        clean_tokens = tokens
                                                    #name_tag = name_tagger.get_name_tag(table_data[i][j])

                                                    for k, tk in enumerate(tokens):
                                                        tokens_.append(tk)
                                                        clean_tokens_.append(clean_tokens[k])
                                                        rows_.append(i + 1)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])
                                                        #name_tags_.append(name_tag)

                                                        if k >= 50:
                                                            break

                                                    if len(tokens) > 50 and str(table_data[i][j]).find(
                                                            '[/answer]') != -1:
                                                        tokens_.append('[/answer]')
                                                        rows_.append(i)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])
                                                        #name_tags_.append(name_tag)
                                        print(clean_tokens_)
                                        start_idx = -1
                                        end_idx = -1

                                        tokens = []
                                        clean_tokens = []
                                        rows = []
                                        cols = []
                                        ranks = []
                                        segments = []
                                        name_tags = []

                                        for tk in query_tokens:
                                            tokens.append(tk)
                                            clean_tokens.append(tk)
                                            rows.append(0)
                                            cols.append(0)
                                            ranks.append(0)
                                            segments.append(0)
                                            name_tags.append(0)

                                        for j, tk in enumerate(tokens_):
                                            if tk == '[answer]':
                                                start_idx = len(tokens)
                                            elif tk == '[/answer]':
                                                end_idx = len(tokens) - 1
                                            else:
                                                tokens.append(tk)
                                                clean_tokens.append(clean_tokens_[j])
                                                rows.append(rows_[j] + 1)
                                                cols.append(cols_[j] + 1)
                                                ranks.append(ranks_[j])
                                                segments.append(1)
                                                #name_tags.append(name_tags_[j])

                                        ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)

                                        #print(tokens)
                                        #input()

                                        max_length = 512

                                        length = len(ids)
                                        if length > max_length:
                                            length = max_length

                                        input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        input_mask = np.zeros(shape=[1, max_length], dtype=np.int32)

                                        segments_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        ranks_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        cols_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        rows_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        names_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)

                                        count = 0

                                        for j in range(length):
                                            input_ids[count, j] = ids[j]
                                            segments_has_ans[count, j] = segments[j]
                                            cols_has_ans[count, j] = cols[j]
                                            rows_has_ans[count, j] = rows[j]
                                            ranks_has_ans[count, j] = ranks[j]
                                            input_mask[count, j] = 1
                                            #names_has_ans[count, j] = name_tags[j]

                                        feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                                                     self.input_segments: segments_has_ans,
                                                     self.input_names: names_has_ans,
                                                     self.input_rankings: ranks_has_ans,
                                                     self.input_rows: rows_has_ans, self.input_cols: cols_has_ans}

                                        if input_ids.shape[0] > 0:

                                            probs_start, probs_stop = \
                                                sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                                            probs_start = np.array(probs_start, dtype=np.float32)
                                            probs_stop = np.array(probs_stop, dtype=np.float32)

                                            for j in range(input_ids.shape[0]):
                                                for k in range(1, input_ids.shape[1]):
                                                    probs_start[j, k] = 0
                                                    probs_stop[j, k] = 0

                                                    if input_ids[j, k] == 3:
                                                        break

                                            self.chuncker.get_feautre(question)

                                            prob_scores = []
                                            c_scores = []

                                            for j in range(input_ids.shape[0]):
                                                # paragraph ranking을 위한 score 산정기준
                                                # score2 = ev_values[j, 0]
                                                score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                                                prob_scores.append(score2)
                                                #c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                                            if True:
                                                for j in range(input_ids.shape[0]):
                                                    probs_start[j, 0] = -999
                                                    probs_stop[j, 0] = -999

                                                # CLS 선택 무효화

                                                prediction_start = probs_start.argmax(axis=1)
                                                prediction_stop = probs_stop.argmax(axis=1)

                                                answers = []
                                                scores = []
                                                candi_scores = []

                                                for j in range(input_ids.shape[0]):
                                                    answer_start_idx = prediction_start[j]
                                                    answer_stop_idx = prediction_stop[j]

                                                    if cols_has_ans[0, answer_start_idx] != cols_has_ans[0, answer_stop_idx]:
                                                        answer_stop_idx2 = answer_stop_idx
                                                        answer_stop_idx = answer_start_idx
                                                        answer_start_idx2 = answer_stop_idx2

                                                        for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_start_idx]:
                                                                answer_stop_idx = k
                                                            else:
                                                                break

                                                        for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_stop_idx2]:
                                                                answer_start_idx2 = k
                                                            else:
                                                                break

                                                        prob_1 = probs_start[0, answer_start_idx] + \
                                                                 probs_stop[0, answer_stop_idx]

                                                        prob_2 = probs_start[0, answer_start_idx2] + \
                                                                 probs_stop[0, answer_stop_idx2]

                                                        if prob_2 > prob_1:
                                                            answer_start_idx = answer_start_idx2
                                                            answer_stop_idx = answer_stop_idx2

                                                    score = probs_start[j, answer_start_idx]
                                                    scores.append(score * 1)
                                                    candi_scores.append(score * 1)

                                                    if answer_start_idx > answer_stop_idx:
                                                        answer_stop_idx = answer_start_idx + 15
                                                    if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                                        for k in range(answer_start_idx, input_ids.shape[1]):
                                                            if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                                                answer_stop_idx = k
                                                                break

                                                    answer = ''

                                                    if answer_stop_idx + 1 >= input_ids.shape[1]:
                                                        answer_stop_idx = input_ids.shape[1] - 2

                                                    probs1 = probs_start[0, answer_start_idx]
                                                    probs2 = probs_stop[0, answer_stop_idx]
                                                    new_start_idx = answer_start_idx
                                                    new_stop_idx = answer_stop_idx

                                                    for k in range(answer_start_idx, answer_stop_idx + 1):
                                                        if k < 512 - 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k + 1]:
                                                                probs1 += probs_stop[0, k]
                                                                new_stop_idx = k
                                                                break

                                                    for k in list(
                                                            reversed(range(answer_start_idx, answer_stop_idx + 1))):
                                                        if k > 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                                                probs2 += probs_start[0, k]
                                                                new_start_idx = k
                                                                break

                                                    if probs1 > probs2:
                                                        answer_stop_idx = new_stop_idx
                                                    else:
                                                        answer_start_idx = new_start_idx

                                                    for a_i, k in enumerate(
                                                            range(answer_start_idx, answer_stop_idx + 1)):
                                                        if a_i > 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                                                break
                                                        answer += str(clean_tokens[k])

                                                    answers.append(answer.replace(' ##', ''))

                                            if len(answers) > 0:
                                                answer_candidates = []
                                                candidates_scores = []

                                                for _ in range(1):
                                                    m_s = -99
                                                    m_ix = 0

                                                    for q in range(len(scores)):
                                                        if m_s < scores[q]:
                                                            m_s = scores[q]
                                                            m_ix = q

                                                    answer_candidates.append(answer_re_touch(answers[m_ix]))
                                                    candidates_scores.append(candi_scores[m_ix])
                                                    # print('score:', scores[m_ix])
                                                    # scores[m_ix] = -999

                                                a1 = [0]
                                                a2 = [0]

                                                for a_c in answer_candidates:
                                                    if a_c.find('<table>') != -1:
                                                        continue

                                                    a1.append(
                                                        exact_match_score(prediction=a_c, ground_truth=answer_text))
                                                    a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                                                for q in range(len(queries)):
                                                    if queries[q].strip() == question.strip():
                                                        code = codes[q]

                                                        if code == 1 or code == 2:
                                                            em_total1 += max(a1)
                                                            f1_total1 += max(a2)
                                                            epo1 += 1
                                                        else:
                                                            em_total2 += max(a1)
                                                            f1_total2 += max(a2)
                                                            epo2 += 1

                                                if epo1 > 0 and epo2 > 0:
                                                    print('EM1:', em_total1 / epo1)
                                                    print('F11:', f1_total1 / epo1)
                                                    print()
                                                    print('EM2:', em_total2 / epo2)
                                                    print('F12:', f1_total2 / epo2)

                                                em_total += max(a1)
                                                f1_total += max(a2)
                                                epo += 1

                                                for j in range(input_ids.shape[0]):
                                                    check = 'None'
                                                    answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                                                    f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                                                   ground_truth=answer_text)

                                                    text = answers[j]

                                                    print('score:', scores[j], check, type, 'F1:',
                                                          f1_, ' , ',
                                                          text.replace('\n', ' '))
                                                print(table_text.replace('\n', ''))
                                                print('question:', question)
                                                print('answer:', answer_text)
                                                print('EM:', em_total / epo)
                                                print('F1:', f1_total / epo)
                        