import numpy as np
import json

from transformers import AutoTokenizer
import pos_augmented_tokenizer

### tokenizer: we made
#tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
tokenizer = pos_augmented_tokenizer.TokenizerAugmentedPOS()

### declare variable
total_num = 40000
max_length = 64
input_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
token_type_ids = np.zeros(shape=[total_num, max_length], dtype=np.int32)
label = np.zeros(shape=[total_num, 3])

pos_attention_label = np.zeros(shape=[total_num, max_length, max_length], dtype=np.int32)

data = json.load(open('/data/KLUE-baseline/data/klue_benchmark/klue-sts-v1.1/klue-sts-v1.1_train.json', 'r', encoding='utf-8'))
label_dict = {1: 1, 0: 0}

idx = 0
tokenizer.max_length = max_length

for data_dict in data:
    # print(data_dict)
    premise = data_dict['sentence1']
    hypothesis = data_dict['sentence2']
    label_text = data_dict['labels']['binary-label']
    print(label_text)

    if idx % 100 == 0:
        print(idx, len(tokenizer.tokenize(premise)), len(tokenizer.tokenize(hypothesis)), label_dict[label_text])

    try:
        tokens = ['[CLS]']

        seq_tokens, attention_matrix_premise = tokenizer.make_attention_matrix(
            premise,
            add=len(tokens))

        tokens.extend(seq_tokens)
        tokens.append('[SEP]')

        segments = [0] * len(tokens)

        seq_tokens, attention_matrix_hypothesis = tokenizer.make_attention_matrix(
            hypothesis,
            add=len(tokens))
        tokens.extend(seq_tokens)
        segments.extend([1] * len(tokens))
    except:
        continue

    ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

    length = len(ids)
    if length > max_length:
        length = max_length

    for j in range(length):
        input_ids[idx, j] = ids[j]
        token_type_ids[idx, j] = segments[j]
    label[idx, label_dict[label_text]] = 1

    pos_attention_label[idx] = attention_matrix_premise + attention_matrix_hypothesis
    idx += 1
    # if idx == 2000:
    #     break

np.save('sts_data/input_ids', input_ids[0:idx])
np.save('sts_data/token_type_ids', token_type_ids[0:idx])
np.save('sts_data/label', label[0:idx])
np.save('sts_data/pos_attention', pos_attention_label[0:idx])




