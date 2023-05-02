from konlpy.tag import Mecab as Mecab
import re


pattern = '(<([^>]+)>)'


class Chuncker:
    def __init__(self):
        self.tagger = Mecab()
        self.Bi_charcter_feature = []

    def get_feautre(self, query):
        self.Bi_charcter_feature = []
        query = re.sub(pattern=pattern, repl='', string=query)
        # print(query)

        TKs = self.tagger.morphs(query)
        pos = self.tagger.pos(query)

        for t, TK in enumerate(TKs):
            if pos[t][1][0] == 'N':
                self.Bi_charcter_feature.append(str(TK))

        # print(self.Bi_charcter_feature)

        #print(self.Bi_charcter_feature)

    def get_chunk_score(self, paragraph):
        score = 0

        for ch_feat in self.Bi_charcter_feature:
            if paragraph.find(ch_feat) != -1:
                score += 1

        if len(self.Bi_charcter_feature) == 0:
            return 0

        return score / len(self.Bi_charcter_feature)
