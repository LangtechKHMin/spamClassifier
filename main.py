from preproc import csv_open, SpamParser, txt2list
import os, warnings

warnings.filterwarnings(action='ignore')

noise_dict1 = {'[£\:\#\*\,]':'', 'Dear \w+':'', "'m":' am', "'ll":' will', "'t" : ' not',"'d" : ' would', "( ur | UR )":' your ', " u " : ' you ', '\w*\d+\w*\d*' : 'numbers'}
noise_dict2 = {'[\.?!]$':'', 'numbers':''}
spam_pair = {"give":"money","buy":"product","get":"money","show":"cash","provide":"free", "call":"number","check":"money"}

print("[1] DATA PREPARED")
spam, ham = csv_open("spamraw.csv", ["type","text"],["spam","ham"])
#os.system("python spam_reader.py")
test = txt2list("spam_test.txt")

# spam = spam[0:10]
# ham = ham[0:10]
# test = test[0:10]

test = [t for t in test if len(t) > 10]
print("TEST : {}".format(len(test)))
parser = SpamParser(spam, ham)
test_data = SpamParser(test, [])
'''
print("[2] FEATURE EXTRACTED for TRAIN DATA")

#print("Spam Example : {0}".format(parser.spam[0]))
#print("Ham Example : {0}".format(parser.ham[0]))

n_spam, n_ham, ns_spam, ns_ham = parser.token_number(noise_dict1, noise_dict2, False)


rspam = parser.text2sim(parser.spam, noise_dict1, noise_dict2, spam_pair)
rham = parser.text2sim(parser.ham, noise_dict1, noise_dict2, spam_pair)

print("[3] FEATURE EXTRACTED for TEST DATA")

# test data feature Extraction
test_data = SpamParser(test, [])
n_test, _, ns_test, _ = test_data.token_number(noise_dict1, noise_dict2, False)

rtest = test_data.text2sim(test_data.spam, noise_dict1, noise_dict2, spam_pair)
#############################################################################

n_whole = n_spam+n_ham+n_test
n_max, n_min = max(n_whole), min(n_whole)
ns_whole = ns_spam + ns_ham + ns_test
ns_max, ns_min = max(ns_whole), min(ns_whole)

n_spam, n_ham, n_test = parser.min_max_scaler(n_spam, n_max, n_min), parser.min_max_scaler(n_ham, n_max, n_min), test_data.min_max_scaler(n_test, n_max, n_min)
ns_spam, ns_ham, ns_test = parser.min_max_scaler(ns_spam, ns_max, ns_min), parser.min_max_scaler(ns_ham, ns_max, ns_min), test_data.min_max_scaler(ns_test, ns_max, ns_min)

buf1 = list(zip(rspam[0],rspam[1],rspam[2],rspam[3],rspam[4],rspam[5],rspam[6]))
buf2 = list(zip(rham[0],rham[1],rham[2],rham[3],rham[4],rham[5],rham[6]))
buf3 = list(zip(rtest[0],rtest[1],rtest[2],rtest[3],rtest[4],rtest[5],rtest[6]))

mrspam, mrham, mrtest = [max(s) for s in buf1], [max(s) for s in buf2], [max(s) for s in buf3]

r_whole = mrspam+mrham+mrtest
r_max, r_min = max(r_whole), min(r_whole)

Rmrspam, Rmrham, Rmrtest = parser.min_max_scaler(mrspam, r_max, r_min), parser.min_max_scaler(mrham, r_max, r_min), test_data.min_max_scaler(mrtest, r_max, r_min)

#############################################################################
# FEATURE DATA WRITE #
# DATA : n_spam  ns_spam  mrspam  Rmrspam  rspam[0]  rspam[1]  rspam[2] rspam[3] rspam[4] rspam[5] rspam[6]
nSPAM, nHAM, nTEST = len(n_spam), len(n_ham), len(n_test)
with open("data.csv","wt") as f:
    for n in range(nSPAM):
        f.writelines(str(n_spam[n])+","+str(ns_spam[n])+","+str(mrspam[n])+","+str(Rmrspam[n])+","+str(rspam[0][n])+","+str(rspam[1][n])+","+str(rspam[2][n])+","+str(rspam[3][n])+","+str(rspam[4][n])+","+str(rspam[5][n])+","+str(rspam[6][n])+","+str(1.)+"\n")
    for n in range(nHAM):
        f.writelines(str(n_ham[n])+","+str(ns_ham[n])+","+str(mrham[n])+","+str(Rmrham[n])+","+str(rham[0][n])+","+str(rham[1][n])+","+str(rham[2][n])+","+str(rham[3][n])+","+str(rham[4][n])+","+str(rham[5][n])+","+str(rham[6][n])+","+str(0.)+"\n")
    for n in range(nTEST):
        f.writelines(str(n_test[n])+","+str(ns_test[n])+","+str(mrtest[n])+","+str(Rmrtest[n])+","+str(rtest[0][n])+","+str(rtest[1][n])+","+str(rtest[2][n])+","+str(rtest[3][n])+","+str(rtest[4][n])+","+str(rtest[5][n])+","+str(rtest[6][n])+","+str(1.)+"\n")
    f.close()
'''
#############################################################################
# FEATURE DATA READ #
import numpy as np
xy = np.loadtxt('data.csv', delimiter = ',', dtype = np.float32)

S, H, T = 747, 2406, 2406+2189
# 2406 / 2406
n_spam, ns_spam, mrspam, Rmrspam, rspam = xy[:S,0], xy[:S,1], xy[:S,2], xy[:S,3], xy[:S, 4:-1]
n_ham, ns_ham, mrham, Rmrham, rham = xy[747:747+H,0], xy[747:747+H,1], xy[747:747+H,2], xy[747:747+H,3], xy[747:747+H, 4:-1]
n_test, ns_test, mrtest, Rmrtest, rtest = xy[3153:3153+T,0], xy[3153:3153+T,1], xy[3153:3153+T,2], xy[3153:3153+T,3], xy[3153:3153+T, 4:-1]

rspam, rham, rtest = [list(n) for n in rspam], [list(n) for n in rham], [list(n) for n in rtest]

from nltk import sent_tokenize, word_tokenize
T0, H, T = spam[1], ham[0], test[0]
T1 = parser.noise_converter(noise_dict1,T0)
T2 = sent_tokenize(T1)
S01, S02, S03 = T2[0], T2[1], T2[2]
for n in range(len(T2)):
    parser.noise_converter(noise_dict2, T2[n])
S1, S2, S3 = T2[0], T2[1], T2[2]
SWP1, SWP2, SWP3 = parser.sent_word_pos(S1)
P1 = parser.prop_analyser(SWP3, SWP1)
A, _, C = parser.sent_word_pos(S2)
P2 = parser.prop_analyser(C, A)
A, _, C = parser.sent_word_pos(S2)
P3 = parser.prop_analyser(C, A)
Props = [P1,P2,P3]
sim1 = parser.prop2sim(Props, "give", "money")
R = parser.text2sim([T0], noise_dict1, noise_dict2, spam_pair)



print("\n[1] Raw Text Example:\n{}".format(T0))
print("\n[2] Noise Removal 1:\n{}".format(parser.noise_converter(noise_dict1,T1)))
print("\n[3] Result of Sentence Tokenizer:\n{}\n{}\n{}".format(S01,S02,S03))
print("\n[4] Noise Removal 2:\n<1> {}\n<2> {}\n<3> {}".format(S1,S2,S3))
print("\n[5-1] Sentence to Word Tokens and POS tagged Set (Word Tokens) :\n{}".format(SWP1))
print("\n[5-2] Sentence to Word Tokens and POS tagged Set (POS Tagged) :\n{}".format(SWP3))
print("\n[6] Propositions from Word Tokens and POS tagged set :\n{}".format(P1))
print("\n[7] For Spam Pair [GIVE MONEY], Calculation of Word Similarity and highest values :\n{}".format(sim1))
print("\n[8] For 7 spam pairs, the results of similarities:\n{}".format(R))


#############################################################################
print("<TOKEN>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(n_spam))
HAM = list(zip(n_ham))
TEST = list(zip(n_test))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])


print("<SENT>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(ns_spam))
HAM = list(zip(ns_ham))
TEST = list(zip(ns_test))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])



print("<PROPs>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(rspam)
HAM = list(rham)
TEST = list(rtest)
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<TOKEN + SENT>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(n_spam, ns_spam))
HAM = list(zip(n_ham, ns_ham))
TEST = list(zip(n_test, ns_test))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])



print("<TOKEN + PROPs>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = np.hstack([list(zip(n_spam)), rspam])
HAM = np.hstack([list(zip(n_ham)), rham])
TEST = np.hstack([list(zip(n_test)), rtest])
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<SENT + PROPs>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = np.hstack([list(zip(ns_spam)), rspam])
HAM = np.hstack([list(zip(ns_ham)), rham])
TEST = np.hstack([list(zip(ns_test)), rtest])
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<TOKEN + SENT + PROPs>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = np.hstack([list(zip(n_spam)),list(zip(ns_spam)), rspam])
HAM = np.hstack([list(zip(n_ham)),list(zip(ns_ham)) ,rham])
TEST = np.hstack([list(zip(n_test)),list(zip(ns_test)) ,rtest])
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<MAX PROP - not MinMaxScaler>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(mrspam))
HAM = list(zip(mrham))
TEST = list(zip(mrtest))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<MAX PROP - use MinMaxScaler>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(Rmrspam))
HAM = list(zip(Rmrham))
TEST = list(zip(Rmrtest))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<TOKEN + MAX PROP - no MinMaxScaler>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(n_spam, mrspam))
HAM = list(zip(n_ham, mrham))
TEST = list(zip(n_test, mrtest))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])

print("<TOKEN + MAX PROP - use MinMaxScaler>")
print("[4] PREPROCESSING CLASSIFIER")
SPAM = list(zip(n_spam, Rmrspam))
HAM = list(zip(n_ham, Rmrham))
TEST = list(zip(n_test, Rmrtest))
print("[5] SVM CLASSIFIER")
spam_result = parser.svm_classifier(SPAM, HAM)
parser.ERR_calculator(spam_result, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
spam_result = parser.svm_classifier(SPAM, HAM, test_set = TEST)
parser.ERR_calculator(spam_result, [0 for n in range(2406)]+[1 for n in range(2189)])
print("[6] LR CLASSIFIER")
result = parser.logistic_regression(SPAM, HAM)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [1 for n in range(len(SPAM))]+[0 for n in range(len(HAM))])
result = parser.logistic_regression(SPAM, HAM, test_set = TEST)
RESULT = []
for r in result:
    if r[0] >= 0.5:
        RESULT.append(1)
    else:
        RESULT.append(0)
parser.ERR_calculator(RESULT, [0 for n in range(2406)]+[1 for n in range(2189)])
