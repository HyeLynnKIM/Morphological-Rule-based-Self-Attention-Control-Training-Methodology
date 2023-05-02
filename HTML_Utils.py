import re
import numpy as np
import math

h_tag_list = ['<h1>', '<h2>', '<h3>', '<h4>', '<h5>', '<h6>', '<h7>']
h_tag_list2 = ['</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</h7>']

tag_list = ['<h>', '</h>', '<p>', '</p>']
tag2replace = [' [h]', ' [/h]', ' [p]', ' [/p]']

#tail_text = open('tail.txt', 'r', encoding='utf-8').read()


def pre_process_document(doc, a_token1, a_token2, answer_setting=False, line_combine=False):
    cut_list = ['[a]', '원본 주소', '외부', '개인', '둘러', '위키']

    doc = tag2tag_token(document=doc)
    doc, table_list = tag2token_table(document=doc, open_tag='<table>', close_tag='</table>', token='[table]', token2='[/table]',
                          td_setting=False, answer_setting=answer_setting)
    doc = tag2token_list(document=doc, open_tag='<ul>', close_tag='</ul>', token='[list]', token2='[/list]',
                         answer_setting=answer_setting)
    doc = tag_remove(doc, '<', '>')

    lines = doc.split('\n')
    doc = ''
    for line in lines:
        have_to_cut = False

        for cl in cut_list:
            if len(lines) > len(cl):
                if line[0: len(cl)] == cl:
                    have_to_cut = True

        lc = '\n'
        if line_combine is True:
            lc = ' '

        if have_to_cut is False and len(line) > 20:
            doc += '\n' + line
        else:
            doc += lc + line

    doc = line_space_remove(document=doc)

    ####################################

    doc = doc.replace('\t', ' [answer] ')
    doc = doc.replace('\a', ' [/answer] ')
    doc = doc.replace('[[', '[')
    doc = doc.replace(']]', ']')

    ####################################

    doc = doc.replace('\n [/th]', ' [/th]')
    doc = doc.replace('\n[/p]', ' [/p]')
    doc = space_remove(document=doc)
    doc = doc.replace('\n\n', '\n')
    doc = doc.replace('', '')

    ####################################
    text = '[list] [li] 로그인하지 않음'
    doc = cut_off(document=doc, cut_word=text)

    return doc, table_list


def pre_process_document_(doc, answer_setting):
    cut_list = ['']

    doc = tag2token_table(document=doc, open_tag='<table>', close_tag='</table>', token='[table]', td_setting=False,
                          answer_setting=answer_setting)
    doc = tag2token_list(document=doc, open_tag='<ul>', close_tag='</ul>', token='[list]', token2='[/list]',
                         answer_setting=answer_setting)
    doc = tag2tag_token(document=doc)
    doc = tag_remove(doc, '<', '>')

    lines = doc.split('\n')
    doc = ''
    for line in lines:
        have_to_cut = False

        for cl in cut_list:
            if len(lines) > len(cl):
                if lines[0: len(cl)] == cl:
                    have_to_cut = True

        if have_to_cut is False:
            if len(line) > 25:
                doc += '\n\n'
            doc += ' ' + line

    doc = line_space_remove(document=doc)

    doc = doc.replace('\n [/th]', ' [/th]')
    doc = doc.replace('\n[/p]', ' [/p]')
    doc = space_remove(document=doc)
    doc = doc.replace('\n\n', '\n')
    doc = doc.replace(tail_text, '')

    text = '[list] [li] 로그인하지 않음'
    doc = cut_off(document=doc, cut_word=text)

    return doc


def wrap_text(document, open_tags, close_tags):
    open_count = 0

    result = ''

    for i in range(len(document)):
        for open_tag in open_tags:
            if i + len(open_tag) < len(document):
                if document[i] == open_tag[0] and document[i + 1] == open_tag[1] \
                        and document[i + len(open_tag) - 1] == open_tag[-1] \
                        and document[i + len(open_tag) - 2] == open_tag[len(open_tag) - 2]:
                    open_count += 1

        if open_count > 0:
            result += document[i]

        for close_tag in close_tags:
            if i > len(close_tag):

                if document[i] == close_tag[-1] and document[i - 1] == close_tag[len(close_tag) - 2] \
                        and document[i - len(close_tag) + 1] == close_tag[0] \
                        and document[i - len(close_tag) + 2] == close_tag[1]:
                    open_count -= 1
                    result += '\n'

    return result


def p_tag2token(document, open_tag, close_tag, token, token2, answer_setting=True):
    result = ''

    is_opened = False

    for i in range(len(document)):
        if answer_setting is True:
            if document[i] == '\t':
                result += '[answer]'
            if document[i] == '\a':
                result += '[/answer]'

        if is_opened is False:
            if i + len(open_tag) + 1 < len(document):
                if document[i] == '<' and document[i - 1 + len(open_tag)] == '>' and \
                        document[i + 1] == open_tag[1]:
                    is_opened = True
                    result += token

        if is_opened is True:
            if i > len(close_tag) + 1:
                if document[i - len(close_tag)] == '<' and document[i - 1] == '>' and \
                        document[i - len(close_tag) + 1] == close_tag[1] and \
                        document[i - len(close_tag) + 2] == close_tag[2]:
                    is_opened = False
                    result += token2

    return result


def cut_off(document, cut_word):
    result = ''

    cut = False

    for i in range(len(document)):
        if i + len(cut_word) < len(document):
            if document[i] == cut_word[0] and document[i - 1 + len(cut_word)] == cut_word[-1] and \
                    document[i + 1] == cut_word[1] and document[i - 2 + len(cut_word)] == cut_word[len(cut_word) - 2]:
                return result

        if cut is False:
            result += document[i]

    return result


def space_remove(document):
    for i in range(20):
        document = document.replace('   ', '  ')
        document = document.replace('  ', ' ')

    return document


def line_space_remove(document):
    for i in range(20):
        document = document.replace('\n\n\n', '\n\n')

    document = document.replace('\n\n\n', '\n\n')

    return document


def answer_re_touch(document):
    tags = ['[list]', '[/list]', '[li]', '[/li]']
    tags2 = ['<list>', '</list>', '<li>', '</li>']

    for i in range(len(tag_list)):
        document = document.replace(tag2replace[i], tag_list[i])
    for i in range(len(tags)):
        document = document.replace(tags[i], tags2[i])

    return document


def tag2tag_token(document):
    for i in range(len(tag_list)):
        document = document.replace(tag_list[i], tag2replace[i])
    return document


def normalize_h_tag(document):
    for h_tag in h_tag_list:
        document = document.replace(h_tag, '<h>')
    for h_tag in h_tag_list2:
        document = document.replace(h_tag, '</h>')

    return document


def tag2token_list(document, open_tag, close_tag, token, token2, answer_setting=False):
    table_list = []

    result = ''
    table_str = ''

    is_opened = False
    li_opened = False

    for i in range(len(document)):
        if answer_setting is True:
            if document[i] == '\t':
                result += '[answer]'
            if document[i] == '\a':
                result += '[/answer]'

        if is_opened is False:
            if i + len(open_tag) + 1 < len(document):
                if document[i] == '<' and document[i - 1 + len(open_tag)] == '>' and \
                        document[i + 1] == open_tag[1]:
                    is_opened = True
                    result += token

        if is_opened is True:
            if i > len(close_tag) + 1:
                if document[i - len(close_tag)] == '<' and document[i - 1] == '>' and \
                        document[i - len(close_tag) + 1] == close_tag[1] and \
                        document[i - len(close_tag) + 2] == close_tag[2]:
                    is_opened = False
                    result += token2

                    table_list.append(table_str)
                    table_str = ''

        if is_opened is False:
            result += document[i]
        else:
            table_str += document[i]

            if i > 5 and i + 5 < len(document):
                if document[i - len('<li>')] == '<' and document[i - 1] == '>' and \
                        document[i - len('<li>') + 2] == '<li>'[2] and li_opened is False:
                    li_opened = True
                    result += ' [li] '

                if document[i] == '<' and document[i - 1 + len('</li>')] == '>' and \
                        document[i + 3] == '</li>'[3] and li_opened is True:
                    li_opened = False
                    result += ' [/li] '

            if li_opened is True:
                result += document[i]
            elif document[i] == '\t' or document[i] == '\a':
                result += document[i]

    return result


def tag2token_table(document, open_tag, close_tag, token, token2, td_setting=False, answer_setting=False):
    p_tag = '<p>'

    table_list = []

    result = ''
    table_str = ''

    table_count = 0

    is_opened = False
    th_opened = False
    td_opened = False

    for i in range(len(document)):
        if document[i] == '\t' or document[i] == '\a':
            result += document[i]

        if answer_setting is True:
            if document[i] == '\t':
                result += '[answer]'
            if document[i] == '\a':
                result += '[/answer]'
            if i + len('[answer]') < len(document):
                if document[i:i + len('[answer]')] == '[answer]':
                    result += '[answer]'

        if i + len(p_tag) < len(document):
            if document[i:i + len(p_tag)] == p_tag:
                table_count = 0
                #print('@@@')
                #input()

        if i + len(open_tag) + 1 < len(document):
            if document[i] == '<' and document[i - 1 + len(open_tag)] == '>' and \
                    document[i + 1] == open_tag[1] and document[i + 2] == open_tag[2]:
                table_count += 1

        if i > len(close_tag) + 1:
            if document[i - len(close_tag):i] == close_tag:
                table_count -= 1

        if is_opened is False:
            if i + len(open_tag) + 1 < len(document):
                if document[i] == '<' and document[i - 1 + len(open_tag)] == '>' and \
                        document[i + 1] == open_tag[1] and document[i + 2] == open_tag[2]:
                    is_opened = True
                    result += token

        if is_opened is True and table_count == 0:
            if i > len(close_tag) + 1:
                if document[i - len(close_tag):i] == close_tag:
                    is_opened = False
                    result += token2

                    #print('@@@', table_str)
                    table_list.append(table_str)
                    table_str = ''

        if is_opened is False:
            result += document[i]
        else:
            table_str += document[i]

            if i > 5 and i + 5 < len(document):
                if document[i - len('<th>')] == '<' and document[i - 1] == '>' and \
                        document[i - len('<th>') + 2] == '<th>'[2] and th_opened is False:
                    th_opened = True
                    result += ' [th] '

                if document[i] == '<' and document[i - 1 + len('</th>')] == '>' and \
                        document[i + 3] == '</th>'[3] and th_opened is True:
                    th_opened = False
                    result += ' [/th] '

            if th_opened is True:
                result += document[i]

            if td_setting is True:
                if i > 5 and i + 5 < len(document):
                    if document[i - len('<td>')] == '<' and document[i - 1] == '>' and \
                            document[i - len('<td>') + 2] == '<td>'[2] and td_opened is False:
                        td_opened = True
                        result += ' [td] '

                    if document[i] == '<' and document[i - 1 + len('</td>')] == '>' and \
                            document[i + 3] == '</td>'[3] and td_opened is True:
                        td_opened = False
                        result += ' [/td] '

                if td_opened is True:
                    result += document[i]

    result = result.replace('\t\t', '\t')
    result = result.replace('\a\a', '\a')
    #print(table_count)
    return result, table_list


def tag2token_table_(document, open_tag, close_tag, token, td_setting=False, answer_setting=False):
    result = ''

    is_opened = False
    th_opened = False

    answer_tag1 = '[answer]'
    answer_tag2 = '[/answer]'

    for i in range(len(document)):
        if answer_setting is True:
            if document[i] == '\t':
                result += '[answer]'
            if document[i] == '\a':
                result += '[/answer]'
        if i + len(answer_tag2) + 1 < len(document):
            if document[i:i + len(answer_tag1)] == answer_tag1:
                result += '[answer]'
            if document[i:i + len(answer_tag2)] == answer_tag2:
                result += '[/answer]'

        if is_opened is False:
            if i + len(open_tag) + 1 < len(document):
                if document[i] == '<' and document[i - 1 + len(open_tag)] == '>' and \
                        document[i + 1] == open_tag[1]:
                    is_opened = True
                    result += token

        if is_opened is True:
            if i > len(close_tag) + 1:
                if document[i - len(close_tag)] == '<' and document[i - 1] == '>' and \
                        document[i - len(close_tag) + 1] == close_tag[1] and \
                        document[i - len(close_tag) + 2] == close_tag[2]:
                    is_opened = False

        if is_opened is False:
            result += document[i]
        else:
            if document[i - len('<th>')] == '<' and document[i - 1] == '>' and \
                    document[i - len('<th>') + 2] == '<th>'[2] and th_opened is False:
                th_opened = True
                result += ' [th] '

            if document[i] == '<' and document[i - 1 + len('</th>')] == '>' and \
                    document[i + 3] == '</th>'[3] and th_opened is True:
                th_opened = False
                result += ' [/th] '

            if th_opened is True:
                result += document[i]

    return result


def tag_remove(document, open_tag, close_tag):
    result = ''

    is_opened = False

    for i in range(len(document)):
        if document[i] == open_tag:
            is_opened = True

        if is_opened is False:
            result += document[i]

        if document[i] == close_tag:
            is_opened = False

    return result


def line_space_process(document, token):
    result = ''

    for i in range(len(document)):
        if document[i - len(token)] == token[0] and document[i - 1] == token[-1] and document[i] == token[0]:
            0
        else:
            result += document[i]

    return result


def table_pre_process(document):
    is_opened = True

    result = ''

    for i in range(len(document)):
        if i > 3:
            if document[i - 3: i] == '<th' and document[i] != '>':
                is_opened = False

        if is_opened is False:
            if document[i] == '>':
                is_opened = True

        if is_opened is True:
            result += document[i]

    return result


def get_table_head_list(document):
    is_opened = False
    valid_heads = True

    result = ''
    table_heads = []
    table_heads_temp = []

    for i in range(len(document)):
        if i > 4:
            if document[i - len('</tr>'): i] == '</tr>':
                if valid_heads is True:
                    if len(table_heads) < len(table_heads_temp):
                        table_heads = table_heads_temp
                table_heads_temp = []

        if i > 3:
            if document[i - 4: i] == '<td>':
                valid_heads = False

        if i > 3:
            if document[i - 4: i] == '<th>':
                is_opened = True

        if is_opened is True:
            if document[i: i + 5] == '</th>':
                is_opened = False
                table_heads_temp.append(result)
                result = ''
            else:
                result += document[i]

    return table_heads


def get_table_data_list(document):
    tr_is_opened = False
    td_is_opened = False

    result = ''
    table_data = []

    is_valid = False

    for i in range(len(document)):
        if document[i] == '\t':
            result += '[answer]\t'

        if document[i: i + 5] == '</tr>':
            if is_valid is True:
                table_data.append(result)
            is_valid = False
            result = ''

        if i > 3:
            if document[i - 4: i] == '<th>':
                td_is_opened = True

        if document[i: i + 5] == '</th>':
            td_is_opened = False
            result += '\t\a'

        if i > 3:
            if document[i - 4: i] == '<td>':
                td_is_opened = True
                is_valid = True

        if document[i: i + 5] == '</td>':
            td_is_opened = False
            result += '\t\a'

        if td_is_opened is True:
            result += document[i]

    return table_data


def table_preprocess(doc):
    is_opened = True

    result = ''

    for i in range(len(doc)):
        if i + 12 < len(doc):
            if doc[i:i + len(' colspan=')] == ' colspan=':
                is_opened = False
                print('@@@')
            if doc[i:i + len(' rowspan=')] == ' rowspan=':
                is_opened = False

        if i > 2:
            if is_opened is False:
                if doc[i - len('\">'):i] == '\">':
                    is_opened = True

        if is_opened is True:
            result += doc[i]

    return result


def pre_process_table_value(doc):
    doc = doc.replace('[a]', '')
    doc = doc.replace('[/a]', '')
    doc = doc.replace('[img]', '')
    doc = doc.replace('<span>', '')
    doc = doc.replace('</span>', '')
    doc = doc.replace('[b]', '')
    doc = doc.replace('[/b]', '')
    doc = doc.replace('<b>', '')
    doc = doc.replace('</b>', '')
    doc = doc.replace('<small>', '')
    doc = doc.replace('</small>', '')
    doc = doc.replace('<abbr>', '')
    doc = doc.replace('</abbr>', '')
    doc = doc.replace('<sup>', '')
    doc = doc.replace('</sup>', '')

    doc = doc.replace('\n', '')

    return doc


def basic_pre_process(text):
    text = text.replace('\a', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')

    return text


def img_tag(text):
    img_list = []

    tag = '<img'

    cnt = 0

    opened = False
    tag_words = ''

    for i in range(len(text) - len(tag)):
        if text[i:i + len(tag)] == tag:
           opened = True
           cnt += 1

        if opened is True:
            tag_words += text[i]
            if text[i] == '>':
                opened = False

                tag_opened = False
                count = 0

                for j in range(i, len(text) - len(tag)):
                    if text[j] == '<':
                        tag_opened = True

                    if tag_opened is True and text[j] == '>':
                        tag_opened = False
                    elif tag_opened is False:
                       tag_words += text[j]
                       count += 1

                    if count == 30:
                        break

                img_list.append(tag_words)

                tag_words = ''

    return cnt, img_list


def check_transpose_table(table_text):
    table_text = str(table_text).replace('\n', '').replace('\t', '').replace('\'', '\"')

    lines = table_text.split('<tr>')
    lines.pop(0)

    head_count = 0
    for i in range(len(lines)):
        if lines[i].find('<th') != -1:
            head_count += 1

    if head_count == len(lines) and head_count > 0:
        return True
    return False


def table_text_process(table_text):
    table_tag = False

    table_text = table_text.replace('<br>', '')
    table_text = table_text.replace('<br/>', '')
    table_text = table_text.replace('<small>', '')
    table_text = table_text.replace('</small>', '')

    table_tag_text = '<table>'
    table_tag_text_ = '<table'

    table_tag_text2 = '</table>'

    pattern = '<[^>]*>'

    double_table = False
    temp = ''

    result = ''

    for i in range(len(table_text)):
        if table_tag is False:
            if table_text[i: i + len(table_tag_text_)] == table_tag_text_:
                table_tag = True
                #print(i)
        else:
            if i + len(table_tag_text) < len(table_text):
                if table_text[i: i + len(table_tag_text)] == table_tag_text:
                    double_table = True
            if i - len(table_tag_text2) > 0:
                if table_text[i - len(table_tag_text2):i] == table_tag_text2:
                    double_table = False

                    temp = re.sub(pattern=pattern, repl='', string=temp)
                    result += temp
                    temp = ''

        if double_table is False:
            result += table_text[i]
        elif table_text[i] == '\t':
            result += '\t'
        else:
            temp += table_text[i]

    return result


def get_table_data(table_text):
    table_text = str(table_text).replace('\n', '').replace('\'', '\"')
    lines = table_text.split('<tr>')
    lines.pop(0)

    head_count = 0
    tranposed_table = False

    for i in range(len(lines)):
        if len(lines[i].split('<th')) > 3:
            head_count = -1
            tranposed_table = False
            break
        if lines[i].find('<th') != -1:
            head_count += 1

    if head_count == len(lines) and head_count > 0:
        tranposed_table = True

    if tranposed_table is True:
        table_text = table_text.replace('<th', '<td').replace('</th', '</td')

    lines = table_text.split('<tr>')
    lines.pop(0)
    lines_arr = []

    for i in range(len(lines)):
        data_arr = []
        lines_arr.append(data_arr)

    for i in range(len(lines_arr)):
        if len(lines[i].split('<th')) < 3:
            TK = lines[i].replace('<th', '<td').replace('</th', '</td').split('<td')
            TK.pop(0)
            #print('@@@@@@@@@@@@')
            #print(TK, len(TK))
            #input()

            for t_d in TK:
                t_d_TK = t_d.replace('</td>', '')
                if len(t_d_TK) > 0:
                    if t_d_TK[0] == '>':
                        t_d_TK = str(t_d_TK[1:len(t_d_TK)])

                if len(t_d_TK) >= 2 or True:
                    word = t_d_TK.split('</tr')[0]

                    if t_d.find('\x07') != -1:
                        word = '[answer]' + word.replace('\x07', '')

                    if t_d_TK.find('rowspan') != -1:
                        word = word.split('>')[1]

                    if t_d_TK.find('colspan') != -1:
                        word = word.split('>')[1]

                        try:
                            epoch = int(t_d_TK.split('\"')[1])
                            for j in range(epoch - 1):
                                lines_arr[i].append(word)
                        except:
                            print('error:', t_d_TK.split('\"'))
                            #input()

                    lines_arr[i].append(word)
                else:
                    lines_arr[i].append('')
    ##############################################################################################################
    counts = []
    TKs = []
    max_length = []

    for i in range(len(lines_arr)):
        TK = lines[i].replace('<th', '<td').replace('</th', '</td').split('<td')
        TK.pop(0)

        TKs.append(TK)
        counts.append(0)
        max_length.append(len(TK))

    try:
        max_length = int(max(max_length))
    except:
        max_length = 0

    for j in range(max_length):
        for i in range(len(lines_arr)):
            if j < len(TKs[i]):
                t_d = TKs[i][j]

                t_d_TK = t_d.replace('</td>', '')
                if len(t_d_TK) > 0:
                    if t_d_TK[0] == '>':
                        t_d_TK = str(t_d_TK[1:len(t_d_TK)])

                if len(t_d_TK) >= 2 and (t_d_TK.find('rowspan') != -1 and t_d_TK.find('colspan') != -1) is not True:
                    word = t_d_TK.split('</tr')[0]
                    if t_d.find('\x07') != -1:
                        word = '[answer]' + word.replace('\x07', '')

                    if t_d_TK.find('rowspan') != -1:
                        word = word.split('>')[1]

                        try:
                            epoch = int(t_d_TK.split('\"')[1])
                            for k in range(epoch - 1):
                                while len(lines_arr[i + k + 1]) < j + counts[i]:
                                    lines_arr[i + k + 1].append('')
                                lines_arr[i + k + 1].insert(j + counts[i], word)
                                counts[i + k + 1] += 1
                        except:
                            print('error!', t_d_TK[0].split('\"'))
                            # input()
    ############################################################################################

    if tranposed_table is True:
        depth = len(lines_arr[-1])

        lines_arr2 = []

        for i in range(1, depth):
            data_arr2 = []

            for line in lines_arr:
                if len(line) == depth:
                    data_arr2.append(line[i])

            lines_arr2.append(data_arr2)

        return lines_arr2

    i = 0
    while True:
        if len(lines_arr) <= i:
            break

        if len(lines_arr[i]) == 0:
            lines_arr.pop(i)
        else:
            i += 1

        if i >= len(lines_arr):
            break

    return lines_arr


def get_table_head(table_text, count_arr=None):
    table_text = str(table_text).replace('\n', '').replace('\t', '').replace('\'', '\"').replace('<span>', '')\
        .replace('</span>', '')

    lines = table_text.split('<tr>')
    lines.pop(0)

    head_count = 0
    tranposed_table = False

    for i in range(len(lines)):
        if len(lines[i].split('<th')) > 3:
            head_count = -1
            tranposed_table = False
            break
        if lines[i].find('<th') != -1:
            head_count += 1

    if head_count == len(lines) and head_count > 0:
        tranposed_table = True

    if tranposed_table is True:
        table_text = table_text.replace('<td', '<th').replace('</td', '</th')

    lines = table_text.split('<tr>')
    lines.pop(0)

    if tranposed_table is True:
        i = 0
        while True:
            if lines[i].find('colspan') != -1:
                lines.pop(i)
            else:
                i += 1

            if i >= len(lines):
                break

    lines_arr = []
    span_arr = []
    #print(lines)
    for i in range(len(lines)):
        data_arr = []
        lines_arr.append(data_arr)

    for i in range(len(lines_arr)):
        if lines[i].find('<th') != -1:
            #if tranposed_table is False:
            #    pattern = r'\<td[^)]*\</td>'
            #    lines[i] = re.sub(pattern=pattern, repl='', string=lines[i])

            TK = lines[i].replace('</th', '</td').replace('<th', '<td').split('<td')
            #print('head', TK)
            #print(lines[i])
            has_span = 0

            for t_d in TK:
                t_d_TK = t_d.replace('</td>', '').split('>')
                #print(t_d_TK)

                if len(t_d_TK) >= 2:
                    word = t_d_TK[1].split('</tr')[0]

                    if t_d_TK[0].find('colspan') != -1:
                        has_span = 1
                        try:
                            epoch = int(t_d_TK[0].split('\"')[1])
                            for j in range(epoch - 1):
                                lines_arr[i].append(word)
                        except:
                            print(t_d_TK[0].split('\"'))
                            #input()

                    lines_arr[i].append(word)
                    #print(lines_arr)
            span_arr.append(has_span)
    ###############################################################################

    counts = []
    span_counts = []
    TKs = []
    max_length = []
    num_heads = []

    for i in range(len(lines_arr)):
        if lines[i].find('<th') != -1:
            heads = lines[i].split('<th')
            num_heads.append(len(heads))

            TK = lines[i].replace('<th', '<td').replace('</th', '</td').split('<td')
            TK.pop(0)

            TKs.append(TK)
            counts.append(0)
            span_counts.append(0)
            max_length.append(len(TK))
        else:
            TKs.append([])
            num_heads.append(0)

    try:
        max_length = int(max(max_length))
    except:
        max_length = 0

    for j in range(max_length):
        for i in range(len(TKs)):
            if j < len(TKs[i]):
                t_d = TKs[i][j]

                t_d_TK = t_d.replace('</td>', '')
                if len(t_d_TK) > 0:
                    if t_d_TK[0] == '>':
                        t_d_TK = str(t_d_TK[1:len(t_d_TK)])

                if len(t_d_TK) >= 2 and (t_d_TK.find('rowspan') != -1 and t_d_TK.find('colspan') != -1) is not True:
                    word = t_d_TK.split('</tr')[0]
                    if t_d.find('\x07') != -1:
                        word = '[answer]' + word.replace('\x07', '')

                    if t_d_TK.find('colspan') != -1:
                        try:
                            epoch = int(t_d_TK.split('\"')[1])
                            span_counts[i] += epoch - 1
                        except:
                            None

                    if t_d_TK.find('rowspan') != -1:
                        word = word.split('>')[1]

                        try:
                            epoch = int(t_d_TK.split('\"')[1])
                            for k in range(epoch - 1):
                                while len(lines_arr[i + k + 1]) < j + counts[i]:
                                    lines_arr[i + k + 1].append('')
                                lines_arr[i + k + 1].insert(j + counts[i] + span_counts[i], word)
                                counts[i + k + 1] += 1
                                #print(word, i + k + 1, span_counts[i])
                        except:
                            print('error!', t_d_TK[0].split('\"'))
                            # input()
    ###############################################################################

    if tranposed_table is True:
        depth = len(lines_arr[-1])

        lines_arr2 = []

        for i in range(1):
            data_arr2 = []

            for line in lines_arr:
                if len(line) == depth:
                    #print(line, depth)
                    data_arr2.append(line[0])

            lines_arr2.append(data_arr2)

        return lines_arr2[-1]

    max_num_heads = max(num_heads)

    if count_arr is not None:
        #count_arr = np.array(count_arr, dtype=np.int32)
        voted_Length = count_arr.argmax()

        if count_arr[voted_Length] > 0:
            for i, head_data in enumerate(list(reversed(lines_arr))):
                if abs(len(lines_arr[i]) - voted_Length) <= 1 and abs(max_num_heads - num_heads[i]) <= 1:
                    return lines_arr[i]

    #for arr in lines_arr:
    #    print('check')
    #    print(len(arr), arr)

    i = 0
    while True:
        if len(lines_arr) == 1:
            break

        if len(lines_arr[i]) > len(lines_arr[i + 1]):
            lines_arr.pop(i + 1)
        else:
            lines_arr.pop(i)

    if len(lines_arr) == 0:
        return False

    return lines_arr[-1]

def abs(num):
    if num < 0:
        return num * -1
    return num


def overlap_table_process(table_text):
    overlap_tables = []

    pattern = '<[^>]*>'

    lines = table_text.split('<tr>')
    result = ''

    close_text = lines.pop(-1)

    table_open = False
    table_text_child = ''

    for i, line in enumerate(lines):
        if i > 0:
            try:
                if line.find('<table>') != -1 and lines[i + 1].find('</table>') != -1:
                    continue
            except:
                None

        if line.find('</table>') != -1:
            if lines[i - 1].find('<table>') != -1:
                text = lines[i - 1] + '<tr>' + lines[i]
                text = text.replace('</table>', '<table>')

                TK = text.split('<table>')

                table_data = re.sub(pattern=pattern, repl='', string=TK[1])
                try:
                    text = TK[0] + table_data.replace('\n', '') + TK[2]
                except:
                    result += line
                result += '<tr>' + text

            else:
                result += '<tr>' + line
        elif i == 0:
            result += line
        else:
            result += '<tr>' + line

    result += '<tr>' + close_text

    return result, overlap_tables

def overlap_table_process2(table_text):
    overlap_tables = []

    pattern = '<[^>]*>'

    lines = table_text.split('<tr>')
    result = ''

    close_text = lines.pop(-1)

    table_open = False
    table_text_child = ''

    for i, line in enumerate(lines):
        if i > 0 and line.find('<table>') != -1 and table_open is False:
            table_open = True
            table_text_child = line
            continue

        if line.find('</table>') != -1:
            if lines[i - 1].find('<table>') != -1:
                text = lines[i - 1] + '<tr>' + lines[i]
                text = text.replace('</table>', '<table>')

                TK = text.split('<table>')

                table_data = re.sub(pattern=pattern, repl='', string=TK[1])
                try:
                    text = TK[0] + table_data + TK[2]
                except:
                    result += line
                result += text

            else:
                result += table_text_child + '<tr>' + line
                child_text = table_text_child + '<tr>' + close_text
                child_text = child_text.replace('</table>', '<table>')
                child_text = '<table>' + child_text.split('<table>')[1] + '</table>'

                overlap_tables.append(child_text)

            table_open = False
            table_text_child = ''
        elif table_open is True:
            table_text_child += '<tr>' + line
            continue
        else:
            result += line

        result += '<tr>'

    if close_text.find('</table>') != -1 and table_open is True:
        result += table_text_child + '<tr>'

        child_text = table_text_child + '<tr>' + close_text
        child_text = child_text.replace('</table>', '<table>')
        child_text = '<table>' + child_text.split('<table>')[1] + '</table>'

        overlap_tables.append(child_text)

    result += close_text

    return result, overlap_tables


def head_process(table_text):
    lines = table_text.split('<tr>')

    for i in range(len(lines)):
        if lines[i].find('<th') != -1 and lines[i].find('<td') != -1:
            lines[i] = lines[i].replace('<th', '<td').replace('</th', '</td')

    result = lines.pop(0)
    for line in lines:
        result += '<tr>' + line

    return result


