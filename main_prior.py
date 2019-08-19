###########################
### SPAM-HAM CLASSIFIER ###
###########################
from preproc import csv_open
# CSV로부터 각 그룹별 리스트를 만들어 주는 부분
spam, ham = csv_open("spamraw.csv", ["type","text"],["spam","ham"])

##########################
### KEYWORD CLASSIFY 1 ### - 추천 단어
##########################
from preproc import classifier_test, sent2word, freq, re_show
# CLASSIFICATION을 위한 정답지 딕셔너리 생성
answer = {}
for s in spam:
    answer[s] = "spam"
for h in ham:
    answer[h] = "ham"
'''

# (Target List, 정답지, "Regular Expression") - FROM 이아름 선생님
classifier_test(spam+ham, answer, "(ppm|PPM)")
classifier_test(spam+ham, answer, "FREE")
classifier_test(spam+ham, answer, "prize")
classifier_test(spam+ham, answer, "STOP")
classifier_test(spam+ham, answer, "URGENT")
classifier_test(spam+ham, answer, "www")
classifier_test(spam+ham, answer, "Nokia")
classifier_test(spam+ham, answer, "(PPM|FREE|prize|STOP|URGENT|www|Nokia)")

classifier_test(spam+ham, answer, "[A-Z]{2,}")
classifier_test(spam+ham, answer, "[A-Z]{3,}")
classifier_test(spam+ham, answer, "[A-Z]{4,}")
classifier_test(spam+ham, answer, "[A-Z]*?\d\d+?[A-Z]*?")
classifier_test(spam+ham, answer, "(([A-Z]*?\d\d+?[A-Z]*?)|PPM|ppm|FREE|prize|STOP|URGENT|www|Nokia|xxx|http)")
classifier_test(spam+ham, answer, "[A-Z]*?\d\d\d+?[A-Z]*?")
classifier_test(spam+ham, answer, "[A-Z]*?\d\d\d(?![\. ])+?[A-Z]*?")


classifier_test(spam+ham, answer, "[A-Z]*?[^\$]\d\d\d(?![\. ])+?[A-Z]*?")

# re_show(spam,"[A-Z]*?\d\d+?[A-Z]*?")

# 대문자 Sequence 4글자 이상인 경우


##########################
### KEYWORD CLASSIFY 2 ### - SPAM의 빈도가 HAM보다 높은 단어
##########################
from preproc import sent2word, freq

spam_word = sent2word(spam)
ham_word = sent2word(ham)

#freq(spam_word)
#freq(ham_word)
# Function word를 제외한 단어들의 빈도가 필요하기에 딕셔너리 look-up을 위한 사전 생성
from preproc import word2pos

spam_sent = sent2word(spam, False)
ham_sent = sent2word(ham, False)
spam_pos = word2pos(spam_sent)
ham_pos = word2pos(ham_sent)

#import nltk
#nltk.help.upenn_tagset()
'''
'''
### VOWEL - V로 태깅된 단어들의 빈도를 보자 ###
from preproc import pos_word
EX_WORD = ["is","am","are","'s","'m","have","had","was","were", "be","been"]

spam_word_verb = pos_word(spam_word, spam_pos, "V", EX_WORD)
ham_word_verb = pos_word(ham_word, ham_pos, "V", EX_WORD)
#print("TOP 10 frequent verbs in SPAM")
#freq(spam_word_verb)
#print("TOP 10 frequent verbs in HAM")
#freq(ham_word_verb)





### NOUN - N로 태깅된 단어들의 빈도를 보자 ###
from preproc import pos_word
EX_WORD = ["is","am","are","'s","'m","have","had","was","were", "be","been"]

spam_word_noun = pos_word(spam_word, spam_pos, "N", EX_WORD)
ham_word_noun = pos_word(ham_word, ham_pos, "N", EX_WORD)
#print("TOP 10 frequent nouns in SPAM")
#freq(spam_word_noun)
#print("TOP 10 frequent nouns in HAM")
#freq(ham_word_noun)
'''



from preproc import sent_tokenizer
from nltk import word_tokenize, pos_tag, sent_tokenize


from nltk.corpus import wordnet as wn
from preproc import wn_tagging, wn_similarity, prop_analyser, spam_classifier

sent_spam = []

f1 = open("sent_token_result.txt","wt", encoding = "utf-16")
for s in spam:
    buf = sent_tokenizer(s)
    sent_spam.append(buf)
    for b in buf:
        f1.writelines(b+"\n")
f1.close()

SPAM_SET = ["purchase product", "give free", "loan money"]





non_error = []
error = []
VAL = []
import re
for s in spam:
    if re.search(r'FREE',s) or re.search(r'URGENT',s):
        continue
    else:

        sent, prop = prop_analyser(s)
        #print(sent[0])
        #print(prop[0])

        result = []
        non_class = []


        for r in prop:
            if r[0] == "spam":
                non_class.append(s)
                result.append("NON")
            else:
                spam_class, value = spam_classifier(r[0],r[1],SPAM_SET)
                result.append(spam_class)
                VAL.append(value)

        nspam, nham, non = 0,0,0
        for res in result:
            if res == "SPAM":
                nspam = nspam+1
            elif res == "HAM":
                nham = nham+1
            elif res == "NON":
                non = non+1


        if nspam > 0:
            # SPAM
            non_error.append(s)
        else:
            error.append(s)

print("Error rate : {0}".format((len(error)/len(spam)*100)))
print("Whole SPAM : {0}".format(len(spam)))
print("SPAM to HAM : {0}".format(len(error)))

print(max(VAL))
print(min(VAL))
print(sum(VAL)/len(VAL))





non_error = []
error = []
VAL = []
import re
for s in ham:
    if re.search(r'FREE',s) or re.search(r'URGENT',s):
        error.append(s)
    else:
        sent, prop = prop_analyser(s)
        #print(sent[0])
        #print(prop[0])

        result = []
        non_class = []


        for r in prop:
            if r[0] == "spam":
                non_class.append(s)
                result.append("NON")
            else:
                spam_class, value = spam_classifier(r[0],r[1],SPAM_SET)
                result.append(spam_class)
                VAL.append(value)

        nspam, nham, non = 0,0,0
        for res in result:
            if res == "SPAM":
                nspam = nspam+1
            elif res == "HAM":
                nham = nham+1
            elif res == "NON":
                non = non+1

        if nspam > 0:
            # SPAM
            error.append(s)
        else:
            pass
            #non_error.append(s)

print("Error rate : {0}".format((len(error)/len(ham)*100)))
print("Whole HAM : {0}".format(len(ham)))
print("HAM to SPAM : {0}".format(len(error)))

print(max(VAL))
print(min(VAL))
print(sum(VAL)/len(VAL))




"""
A = '''k give back my thanks.'''

sent, prop = prop_analyser(A)
print(sent)
print(prop)
"""
# ".." => " "
clitic = {'ur':'you are', 'u' : 'you', "'m'" : 'am'}


################ ['', ''] [None, None] , 문장 -> ham / spam detection이 안되는 경우, spam이 하나라도 있는 경우
# 특정 키워드를 전처리에 삽입하는 경우
