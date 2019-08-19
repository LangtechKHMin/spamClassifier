'''
from sklearn import svm
X = [[0, 0], [1, 1], [0.5,1]]
y = [0, 1, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

T = [[3, 2], [2, 0], [0.5,1]]

for t in T:
    R = clf.predict([list(t)])

    print(R)


A = [(1,1,1),(2,2,2),(3,3,3)]

B = A[0]
for n in range(1,len(A))
    B = list(zip(B,A[n]))
print(B)
'''
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("best"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
