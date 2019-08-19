def csv_open(FILENAME, COL_NAME_LIST, COL_VAL_LIST):
    import pandas as pd
    data = pd.read_csv(FILENAME, dtype = 'unicode')
    A = [data.loc[a,:][COL_NAME_LIST[1]] for a in range(data.shape[0]) if data.loc[a,:][COL_NAME_LIST[0]] == COL_VAL_LIST[0]]
    B = [data.loc[a,:][COL_NAME_LIST[1]] for a in range(data.shape[0]) if data.loc[a,:][COL_NAME_LIST[0]] == COL_VAL_LIST[1]]
    print("SPAM : "+str(len(A))+"\nHAM : "+str(len(B)))
    return A,B

def txt2list(FILENAME):
    f1 = open(FILENAME,"rt")
    txtList = []
    while True:
        line = f1.readline()
        if line != '':
            txtList.append(line[:-1])
        else:
            break
    f1.close()
    return txtList

class SpamParser():
    def __init__(self, spam_set, ham_set):
        self.spam = spam_set
        self.ham = ham_set

    def min_max_scaler(self, data_list, MAX_value = None, MIN_value = None):
        DATA = []
        if MAX_value != None:
            for d in data_list:
                DATA.append((d-MIN_value)/(MAX_value-MIN_value))
        else:
            for d in data_list:
                DATA.append((d-min(data_list))/(max(data_list)-min(data_list)))
        return DATA

    def token_number(self, NOISE_DICT1, NOISE_DICT2 ,MinMaxScaler = True, target = "BOTH"):
        from nltk import word_tokenize, sent_tokenize
        n_spam, n_ham = [], []
        ns_spam, ns_ham = [], []
        for s in self.spam:
            s = self.noise_converter(NOISE_DICT1, s)
            S = sent_tokenize(s)
            ns_spam.append(len(S))
            token_sum = 0
            for s in S:
                s = self.noise_converter(NOISE_DICT2, s)
                tokens, _, _ = self.sent_word_pos(s)
                token_sum += len(tokens)
            n_spam.append(token_sum)
        for s in self.ham:
            s = self.noise_converter(NOISE_DICT1, s)
            S = sent_tokenize(s)
            ns_ham.append(len(S))
            token_sum = 0
            for s in S:
                s = self.noise_converter(NOISE_DICT2, s)
                tokens, _, _ = self.sent_word_pos(s)
                token_sum += len(tokens)
            n_ham.append(token_sum)

        if MinMaxScaler:
            n_spam, n_ham = self.min_max_scaler(n_spam), self.min_max_scaler(n_ham)
            ns_spam, ns_ham = self.min_max_scaler(ns_spam), self.min_max_scaler(ns_ham)

        if target == "TOKEN":
            return n_spam, n_ham
        elif target == "SENT":
            return ns_spam, ns_ham
        else:
            return n_spam, n_ham, ns_spam, ns_ham


    def freq(self, LIST, show = True):
        f_dic = {}
        for x in LIST:
            f_dic[x] = LIST.count(x)
        f_dic = sorted(f_dic.items(), key = lambda x: x[1], reverse = True)
        if show:
            for k, v in f_dic[0:10]:
                print(str(k)+" : "+str(v))
        return f_dic

    def wn_tagging(self, WORD, pos = "V"):
        WORD = WORD.lower()

        ## LEMMATIZATION ##
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        WORD = lemmatizer.lemmatize(WORD)
        ###################

        from nltk.corpus import wordnet as wn
        import re
        SYNS = wn.synsets(WORD)

        if pos == "V":
            for s in SYNS:
                if re.search(r'\.v\.',str(s)):
                    return s
                    break
                else:
                    continue
        elif pos == "N":
            for s in SYNS:
                if re.search(r'\.n\.',str(s)):
                    return s
                    break
                else:
                    continue
        else:
            return 0

    def wn_similarity(self, tg1, tg2, w1, w2, pos1 ="V", pos2 ="N"):
        # (get V cash N - won V camera N)
        from nltk.corpus import wordnet as wn

        TG1 = self.wn_tagging(tg1, pos1)
        TG2 = self.wn_tagging(tg2, pos2)
        W1  = self.wn_tagging(w1, pos1)
        W2  = self.wn_tagging(w2, pos2)

        if not(TG1) or not(TG2) or not(W1) or not(W2):
            return "None"
        else:
            #print("{0} {1} - {2} {3}".format(tg1.upper(),tg2.upper(),w1.upper(),w2.upper()))
            sim_score1 = TG1.path_similarity(W1)
            sim_score2 = TG2.path_similarity(W2)
            #print("{2} score : {0}\n{3} score : {1}\nAVG : {4}".format(sim_score1,sim_score2, w1, w2,(sim_score1+sim_score2)/2))
            return (sim_score1+sim_score2)/2

    def sent_word_pos(self, s):
        from nltk import word_tokenize, pos_tag

        result_s = []
        result_prop = []

        s = s.lower()
        tokenized = word_tokenize(s) # a1 : tokenized sentence [t1, t2, t3, ...]
        token_tag = pos_tag(tokenized) # a2 : pos-tagged tokens [ (t1, p1), (t2, p2), ...]
        tagged = []
        for tag in token_tag:
            tagged.append(tag[1])

        return tokenized, token_tag, tagged

    def prop_analyser(self, TAGGED, TOKENIZED):

        n_predct, next_index, last_code = 0, 0, 0
        PROP = {}
        arg_list = []

        for t in TAGGED:
            next_index += 1
            if t[0] == "V":
                if next_index != len(TAGGED):
                    if TAGGED[next_index][0] != "V":
                        if n_predct == 1:
                            PROP[predicate] = arg_list
                            arg_list = []
                            predicate = TOKENIZED[next_index-1]
                            n_predct = 1
                            # After First Verb
                        else:
                            predicate = TOKENIZED[next_index-1]
                            n_predct = 1
                            # First Verb
                    else:
                        continue
                else:
                    continue

            elif t == "PRP$" and n_predct != 1:
                n_predct = 1
                arg_list = []
                predicate = TOKENIZED[next_index-2]
                arg_list.append(TOKENIZED[next_index-1])

            elif t[0] == "N" and n_predct == 1:
                arg_list.append(TOKENIZED[next_index-1]+"_N")

            else:
                if n_predct == 1:
                    arg_list.append(TOKENIZED[next_index-1])
                else:
                    continue

        if len(arg_list) >= 1:
            PROP[predicate] = arg_list
        else:
            pass

        # LOOP count
        if len(PROP.keys()) == 0:
            return {"null":["null"]}
        else:
            return PROP

    def noise_converter(self, noise_DICT, sent):
        import re
        for k, v in noise_DICT.items():
            rk = re.compile(k)
            sent = re.sub(rk,v,sent)

        return sent
        #for cltc in clitic_DICT

    def sent2prop(self, SENT):
        PROPs = []
        for s in SENT:
            s1,s2,s3 = self.sent_word_pos(s)
            PROP = self.prop_analyser(s3, s1)
            PROPs.append(PROP)
        return PROPs

    def prop2sim(self, PROPs, V, N):
        result = []
        for prop in PROPs:
            values = list(prop.values())[0]
            if len(values)>= 1:
                v = list(prop.keys())[0]
                for val in values:
                    if val[-2:] == "_N":
                        val = val[:-2]
                        sc = self.wn_similarity(V,N,v,val)
                    else:
                        sc = self.wn_similarity(V,N,v,val)

                    if sc != "None":
                        result.append(sc)
            else:
                continue
        return result


    def text2sim(self, TEXT, NOISE_DICT, NOISE_DICT2, SPAM_SET):
        from nltk import word_tokenize, pos_tag, sent_tokenize
        RESULT = [[] for n in range(len(SPAM_SET.keys()))]

        for S in TEXT:
            #1 NOISE REMOVE 1
            S = self.noise_converter(NOISE_DICT, S)

            #2 SENTENCE TOKENIZE
            pre1_S = sent_tokenize(S)

            #3 NOISE REMOVE 2
            for n in range(len(pre1_S)):
                self.noise_converter(NOISE_DICT2, pre1_S[n])

            #4 WORD TOKENIZE & PROPOSITION EXTRACT
            PROPs = self.sent2prop(pre1_S)

            #5 PROPOSITION TO SIMILARITY
            r_count = 0
            for v,n in SPAM_SET.items():
                result = self.prop2sim(PROPs, v, n)
                if result == []:
                    RESULT[r_count].append(0)
                else:
                    RESULT[r_count].append(max(result))
                r_count += 1
        return RESULT

    def svm_classifier(self, A_set, B_set, test_set = ''):
        from sklearn import svm
        import numpy as np

        # train, test data set
        #train_s, train_h = A_set[0:nTrain], B_set[0:nTrain]
        #test_s, test_h = A_set[nTrain:], B_set[nTrain:]

        train_s, train_h = A_set, B_set
        test_s, test_h = A_set, B_set

        if bool(list(test_set)):
            test_x = test_set
        else:
            test_x = np.vstack([test_s,test_h])

        NUM_TRAIN_S, NUM_TRAIN_H = len(train_s), len(train_h)
        NUM_TEST_S, NUM_TEST_H = len(test_s), len(test_h)

        X = np.vstack([train_s, train_h])
        Y = [1 for x in range(NUM_TRAIN_S)]+[0 for x in range(NUM_TRAIN_H)]

        print("TRAIN : {0}".format(len(X)))

        # model fitting
        clf = svm.SVC(gamma='scale')
        clf.fit(X, Y)

        correct = 0
        FEEDBACK = []

        for s in test_x:
            R = clf.predict([list(s)])
            FEEDBACK.append(R[0])

        return FEEDBACK


    def logistic_regression(self, A_set, B_set, test_set = '', epoch = 10000, Learning_Rate = 1e-3):
        import tensorflow as tf
        #tf.random.set_random_seed(777)
        import numpy as np

        train_s, train_h = A_set, B_set
        test_s, test_h = A_set, B_set

        NUM_TRAIN_S, NUM_TRAIN_H = len(train_s), len(train_h)
        NUM_TEST_S, NUM_TEST_H = len(test_s), len(test_h)


        train_x = np.vstack([train_s,train_h])
        s1, s2 = len(train_x), len(train_x[0])

        #print(train_x[0:10])

        set1 = [1. for n in range(s2)]
        set0 = [0. for n in range(s2)]

        train_y = [set1 for x in range(NUM_TRAIN_S)]+[set0 for x in range(NUM_TRAIN_H)]

        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_normal([s2,1]), name = 'weight')
        b = tf.Variable(tf.random_normal([1]), name = 'bias')

        hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = Learning_Rate)
        train = optimizer.minimize(cost)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(epoch):
            sess.run(train, feed_dict={X: train_x, Y: train_y})
            if step % 10000 == 0:
                print(step, sess.run(cost, feed_dict={X: train_x, Y: train_y}), sess.run(W))


        if bool(list(test_set)):
            #print(test_set)
            result = []
            for t in test_set:
                r = sess.run(hypothesis, feed_dict = {X : [t]})[0]
                result.append(r)
        else:
            result = sess.run(hypothesis, feed_dict = {X : train_x})

            #print(result)

        return result
            #if test_set != None:
            #    post_result = sess.run(hypothesis, feed_dict = {X : test_set})


    def ERR_calculator(self, target_set, correct_set):
        ERR, result, one2zero, zero2one = 0, [], 0, 0
        for n in range(len(target_set)):
            if target_set[n] == correct_set[n]:
                result.append(0)
            else:
                if correct_set[n] == 1:
                    ERR += 1
                    result.append(1)
                    one2zero += 1
                else:
                    ERR += 1
                    result.append(1)
                    zero2one += 1
        print("\n*******************************************************************")
        print("ERR : {0} ({1}/{2})".format(ERR/len(result)*100, ERR, len(result)))
        print("FA as 1 : {}\nFA as 0 : {}".format(zero2one, one2zero))
        print("*******************************************************************\n")
        return ERR, result
