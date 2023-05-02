# import Table_Holder
from HTML_Utils import overlap_table_process
import numpy as np
from scipy.stats import rankdata


def extrac_num(word):
    new_word = ''

    for i in range(len(word)):
        if 48 <= ord(word[i]) <= 57:
            new_word += word[i]

    return new_word


def is_num(word):
    if len(word) == 0:
        return False

    num = 0

    for i in range(len(word)):
        if 48 <= ord(word[i]) <= 57:
            num += 1

    if num / len(word) >= 0.5:
        return True

    return False


def printTable(table_data, table_head):
    print('head:', table_head)
    for data in table_data:
        print(data)
    return


def numberToRanking(table_data, table_head):
    # 비교 가능한 칼럼(숫자타입)을 찾는다. 비교가 불가능하면 '0'으로 바꾼다.
    #print('———————비교불가능한 칼럼 0 처리——————')
    i = -1

    rank_data = []

    for col in table_data:
        line_data = []

        i = i + 1
        j = -1
        for item in col:
            j = j + 1
            # 추후에 조건 예를들어 추가 km 같은 단위표현

            if item is not None:
                item = item.replace('[answer]', '')
                item = item.replace('[/answer]', '')

            if item is None:
                line_data.append('-999')
            elif is_num(item) is True:
                line_data.append(extrac_num(item))
            elif item.isnumeric() is True:
                line_data.append(item)
            else:
                line_data.append('-999')

        rank_data.append(line_data)

    rank_data = ([list(map(float, i)) for i in rank_data])
    # https: // stackoverflow.com / questions / 36193225 / numpy - array - rank - all - elements

    return rankdata(rank_data, axis=0, method='min')

"""
table_holder = Table_Holder.Holder()
file = open('test.html', 'r', encoding='utf-8')
table_text = file.read()
table_text, overlap_table_texts = overlap_table_process(table_text=table_text)

table_text = table_text.replace('<th', '<td')
table_text = table_text.replace('</th', '</td')

print(table_text)
input()

table_holder.get_table_text(table_text=table_text)
table_data = table_holder.table_data
table_head = table_holder.table_head

printTable(table_data, table_head)

rank_data = numberToRanking(table_data, table_head)

for data in table_data:
    print(data)

"""