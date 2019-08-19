import re, os
from html.parser import HTMLParser

os.system("cd spam_01 && dir /s /b>../spam_01.scp")
os.system("cd spam_02 && dir /s /b>../spam_02.scp")
os.system("cd spam_03 && dir /s /b>../spam_03.scp")
os.system("cd spam_04 && dir /s /b>../spam_04.scp")


spam_scp = ["spam_01.scp","spam_02.scp","spam_03.scp","spam_04.scp"]

spam_test = []

for spam in spam_scp:
    fin = open(spam,"rt")
    files = []
    while True:
        file = fin.readline()
        if file == '':
            break
        else:
            file = file.split("\\")[-1]
            if file == "cmds\n":
                continue
            else:
                files.append(".//"+spam[:-4]+"//"+file[:-1])
    print(len(files))
    print(files[-1])

    for file in files:
        try:
            f1 = open(file,"rt", encoding = "utf-8")

            content_flag, text = 0, ''
            while True:
                line = f1.readline()
                if line == '':
                    break
                else:
                    if content_flag == 0:
                        if "Content-Transfer-Encoding: base64" in line:
                            break
                        elif "Content-Transfer-Encoding" in line:
                            content_flag += 1
                        else:
                            continue
                    else:
                        if re.search(r'.*?\:',line):
                            continue
                        else:
                            text += line
            t1 = text
            t1 = re.sub(r'[\n\t]','',t1)
            t1 = re.sub(r'=','',t1)
            t1 = re.sub(r'\<.+?\>','',t1)
            t1 = re.sub(r' {2,}',' ',t1)

            spam_test.append(t1)

            f1.close()
        except UnicodeDecodeError as e:
            continue

fout = open("spam_test.txt","wt")
for s in spam_test:
    fout.writelines(s+"\n")

fout.close()
