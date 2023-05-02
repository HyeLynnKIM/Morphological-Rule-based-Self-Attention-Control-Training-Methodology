import numpy as np
from konlpy.tag import Mecab
from transformers import AutoTokenizer

# from Depencency_parsing.baseline.models.lightning_base import *
# from Depencency_parsing.dp_main import *
# from Depencency_parsing.baseline.models.dependency_parsing import *

sentence = '그는 그림 세 개를 샀다.'

class TokenizerAugmentedPOS:
    def __init__(self):
        self.max_length = 512
        self.tagger = Mecab()
        # self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


    def token_pos_align(self, sentence):
        tagger = self.tagger
        tokenizer = self.tokenizer

        idx = 0
        morphs = tagger.morphs(sentence)
        pos = tagger.pos(sentence)
        print(pos)
        # input()

        eujeol_idx = 0

        morphs_eujeol = []
        morphs_e = []

        for m, morph in enumerate(morphs):
            morphs_e.append(pos[m])
            idx += len(morph)

            if idx < len(sentence):
                if sentence[idx] == ' ':
                    morphs_eujeol.append(morphs_e)
                    morphs_e = []
                    eujeol_idx += 1
                    idx += 1
        morphs_eujeol.append(morphs_e)
        # print(morphs_eujeol)
        # input()
        new_pos = []

        for e, eujeol in enumerate(sentence.split(' ')):
            tokens = tokenizer.tokenize(eujeol)
            # print(tokens)

            for token in tokens:
                selected_pos = ''

                try:
                    for morph_pos in morphs_eujeol[e]:
                        if token.replace('##', '') == morph_pos[0]:
                            selected_pos = morph_pos[1]
                except:
                    None

                new_pos.append((token, selected_pos))
                # new_pos.append(selected_pos)
        return new_pos

    def tokenize(self, text):
        sentences = text.replace('\n', ' ').replace('. ', '\n').split('\n')

        tokens = []

        for sentence in sentences:
            sentence = sentence.strip()
            tokens_ = self.tokenizer.tokenize(sentence)
            tokens.extend(tokens_)

        return tokens

    def make_attention_matrix(self, document, add=0):
        # Assume Multi-sentence
        sentences = document.replace('\n', ' ').replace('. ', '\n').split('\n')

        # Make Matrix Format for seq * seq size
        pos_attention_matrix = np.zeros(shape=[self.max_length, self.max_length])

        tokens = []
        power = 1.0
        default = 0.0

        for sentence in sentences:
            # In, Multi-sentences, if there is a space in sentence, split this.
            # But there is no space in sentnece, I think space will be appended in the last.
            sentence = sentence.strip() + '.'

            token_pos = self.token_pos_align(sentence)

            # one token matrix composition
            for p, pos in enumerate(token_pos):
                tokens.append(pos[0])

                # total_length
                for j in range(len(token_pos)):
                    pos_attention_matrix[p + add, j + add] = default

                try:
                    if pos[1] == 'NNB':
                        pos_attention_matrix[p + add, p - 1 + add] = power
                        pos_attention_matrix[p - 1 + add, p + add] = power

                    if pos[1] == 'NNBC':
                        pos_attention_matrix[p - 1 + add, p - 2 + add] = power
                        pos_attention_matrix[p - 2 + add, p - 1 + add] = power

                        pos_attention_matrix[p - 1 + add, p + add] = power
                        pos_attention_matrix[p + add, p - 1 + add] = power

                    # 보격 조사 -> 앞 명사 ### important!!!!
                    if pos[1] == 'JKC':
                        for j in reversed(range(0, p)):
                            if token_pos[j][1][0] == 'N':
                                pos_attention_matrix[p + add, j + add] = power
                                pos_attention_matrix[j + add, p + add] = power
                                break
                        # 보어 == 주어
                        for j in reversed(range(0, p)):
                            if token_pos[j][1] == 'JKS':
                                pos_attention_matrix[p + add, j - 1 + add] = power
                                pos_attention_matrix[j - 1 + add, p + add] = power

                    # if

                    # 관형격 -> 명사 # 80.815,
                    if pos[1] == 'VA':
                        if pos[1] != 'ETM':
                            try:
                                for j in reversed(range(0, p)):
                                    if token_pos[j][1] == 'JKS' or token_pos[j][1] == 'JX':
                                        if token_pos[j-1][1] == 'NNG':
                                            pos_attention_matrix[p + add, j + add] = power
                                            pos_attention_matrix[j + add, p + add] = power
                                            break
                                        elif token_pos[j-1][1] == 'NP':
                                            pos_attention_matrix[p + add, j + add] = power
                                            pos_attention_matrix[j + add, p + add] = power
                                            break
                                        elif token_pos[j-1][1] == 'NNP':
                                            pos_attention_matrix[p + add, j + add] = power
                                            pos_attention_matrix[j + add, p + add] = power
                                            break
                            except:
                                None
                        else:
                            try:
                                for j in range(p + 1, len(token_pos)):
                                    if token_pos[j][1] == 'NNG':
                                        pos_attention_matrix[p - 1 + add, j + add] = power
                                        pos_attention_matrix[j + add, p - 1 + add] = power
                                        break
                                    elif token_pos[j][1] == 'NP':
                                        pos_attention_matrix[p - 1 + add, j + add] = power
                                        pos_attention_matrix[j + add, p - 1 + add] = power
                                        break
                                    elif token_pos[j][1] == 'NNP':
                                        pos_attention_matrix[p - 1 + add, j + add] = power
                                        pos_attention_matrix[j + add, p - 1 + add] = power
                                        break
                                    elif token_pos[j][1] == 'NNB':
                                        pos_attention_matrix[p + add, j + add] = power
                                        pos_attention_matrix[j + add, p + add] = power
                                        break
                            except:
                                None

                    # 부사 -> 부사 혹은 동사
                    if pos[1] == 'MAG':
                        pos_attention_matrix[p + 1 + add, p + add] = power
                        pos_attention_matrix[p + add, p + 1 + add] = power
                        # try:
                        #     for j in range(p + 1, len(token_pos)):
                        #         if token_pos[j][1] == 'MAG':
                        #             pos_attention_matrix[p + add, j + add] = power
                        #             pos_attention_matrix[j + add, p + add] = power
                        #             break
                        #         if token_pos[j][1] == 'VV':
                        #             pos_attention_matrix[p + add, j + add] = power
                        #             pos_attention_matrix[j + add, p + add] = power
                        #             break
                        #         if token_pos[j][1][0] ==
                        # except:
                        #     None

                    if pos[1] == 'VX':
                        for j in range(p+1, len(token_pos)):
                            if token_pos[j][1] == 'VV':
                                pos_attention_matrix[p + add, j + add] = power
                                pos_attention_matrix[j + add, p + add] = power
                                for j in reversed(range(0, p)):
                                    if token_pos[j][1] == 'JX' or token_pos[j][1] == 'JKS':
                                        pos_attention_matrix[p + add, j - 1 + add] = power
                                        pos_attention_matrix[j - 1 + add, p + add] = power

                    # Josa Case
                    if pos[1] in ['JKS', 'JKG', 'JKO', 'JKV', 'JX']:
                        if token_pos[p-1][1][0] == 'N':
                            pos_attention_matrix[p + add, j + add] = power
                            pos_attention_matrix[j + add, p + add] = power

                    # ++
                    if pos[1] == 'JKB':
                        for j in reversed(range(0, p)):
                            if token_pos[j][1] == 'MAG':
                                pos_attention_matrix[p + add, j + add] = power
                                pos_attention_matrix[j + add, p + add] = power
                            else:
                                break

                    # X head ++
                    # if pos[1] == 'XPN':
                    #     pos_attention_matrix[p + add, p + 1 + add] = power
                    #     pos_attention_matrix[p + 1 + add, p + add] = power
                    #
                    # if pos[1][0] == 'X' and pos[1] != 'XPN':
                    #     pos_attention_matrix[p + add, p - 1 + add] = power
                    #     pos_attention_matrix[p - 1 + add, p + add] = power
                except:
                    #print('except!')
                    None

            add += len(token_pos)
        # print(pos_attention_matrix)
        # print(len(pos_attention_matrix))
        # print(len(pos_attention_matrix[0]))
        # input()

        return tokens, pos_attention_matrix

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

# tok = TokenizerAugmentedPOS()
# print(tok.token_pos_align(sentence))
# print(tok.tokenizer(sentence))
# print(tok.make_attention_matrix(sentence))
# print(tok.convert_tokens_to_ids(tok.tokenizer(sentence)))