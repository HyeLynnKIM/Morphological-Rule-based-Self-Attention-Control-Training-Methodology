import numpy as np
import json

from transformers import AutoTokenizer
import pos_augmented_tokenizer
import jsonlines

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

# data = json.load(open('7.prompt_문장평가_실험용.jsonl', 'r', encoding='utf-8'))
# label_dict = {'이 문장은 문법적으로 옳은 문장입니다.': 0, '이 문장은 문법적으로 옳지 않은 문장입니다.':1}
label_dict = {'0': 0 , '중립적인':1, '1':2}
idx = 0
tokenizer.max_length = max_length

with jsonlines.open('7.prompt_문장평가_실험용.jsonl', 'r') as f:
    for data_dict in f:
        premise = data_dict['input']
        label_text = data_dict['output']

        if idx % 100 == 0:
            print(idx, len(tokenizer.tokenize(premise)), label_dict[label_text])

        try:
            tokens = ['[CLS]']

            seq_tokens, attention_matrix_premise = tokenizer.make_attention_matrix(
                premise,
                add=len(tokens))

            tokens.extend(seq_tokens)
            tokens.append('[SEP]')

            segments = [0] * len(tokens)
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

        pos_attention_label[idx] = attention_matrix_premise
        idx += 1
        if idx == 2000:
            break
    print(idx)

np.save('pn_data/input_ids', input_ids[0:idx])
np.save('pn_data/token_type_ids', token_type_ids[0:idx])
np.save('pn_data/label', label[0:idx])
np.save('pn_data/pos_attention', pos_attention_label[0:idx])




