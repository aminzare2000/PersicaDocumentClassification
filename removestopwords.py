from __future__ import unicode_literals
import pandas as pd
from hazm import *
import string
import codecs
import numpy as np
# Python program to Remove all
# digits from a list of string
import re
import chardet


# PATH_TRUEPERSICA = 'D:\\sareh\\amin\\trueTabdelsmall.txt'
PATH_TRUEPERSICA = 'D:\\sareh\\amin\\trueTabdel11000.txt'
PATH_proceedPERSICA = 'proceedPERSICA.txt'
# with open(PATH_TRUEPERSICA, 'rb') as f:
#     result = chardet.detect(f.read())  # or readline if the file is large
# print(result)

df = pd.read_csv(PATH_TRUEPERSICA, sep='\t', encoding='UTF-8', usecols=['Title', 'Body', 'Category2'])
df['proceedTitle'] = ''
df['proceedBody'] = ''
df['proceedTitleBody'] = ''

# print(df.head())
# print(df.describe())
# print(df.info())
# df.groupby('Category2').describe()

# # آدرس فایل مبدا که فایل اصلی در آن قرار دارد
# f = codecs.open('C:\\Users\\Admin\\Desktop\\1.txt', 'r', encoding='utf-8')
#
# # آدرس فایل مقصد که بعد از پیش پردازش محتوا در آن قرار میگیرد
# f2 = codecs.open('C:\\Users\\Admin\\Desktop\\2.txt', 'w', encoding='utf-8')

stopwo = ['آوه', 'تو', 'می باشد', 'باشد', 'باید', 'چرا', 'چون', 'اصلا', 'اصلاً', 'اصولا', 'اصولاً', 'اغلب', 'افسوس',
          'اقل', 'اقليت', 'اكثر', 'الا', 'البته', 'اش', 'اين', 'است', 'اساساً', 'ازش', 'ازاين رو', 'از جمله',
          'از آن پس', 'از', 'احياناً', 'احتمالا', 'اما', 'اتفاقا', 'ان', 'آيا', 'آنچه', 'ولی', 'برای', 'با', 'آنهاست',
          'آنها', 'آنقدر', 'آنگاه', 'آهان', 'آنرا', 'او', 'تو', 'به', 'از', 'آخ', 'آخر', 'آخرها', 'آخه', 'آن ها',
          'آن گاه', 'آن', 'آنان', 'آناني', 'آنجا', 'آنچنان', 'در', 'آنچنان كه']
stop2 = codecs.open('persianstopwords', encoding='utf-8').read().split('\n')


#### Pre Procceess ###
def Preproceed(line):
        # sent = sent_tokenize(line)
        # str = " ".join(sent)

        if line is np.nan:
                return ' '
        words = word_tokenize(line)
        words = remove(words)
        # نرمال سازی
        normalizer = Normalizer()
        words = [normalizer.normalize(w) for w in words]
        # ریشه یابی اسامی
        # stemmer = Stemmer()
        # ریشه یابی افعال
        # lemmatizer = Lemmatizer()
        # words = [stemmer.stem(w) for w in words]
        # words = [lemmatizer.lemmatize(w) for w in words]
        # words=[lemmatizer.lemmatize(w) for w not in stopwo]
        words = [w for w in words if w not in stopwo]
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if w not in stop2]
        str = " ".join(words)
        return str


def remove(list1):
    # pattern = '[0-9]'
    # list = [re.sub(pattern, '', i) for i in list]

    # using filter and lambda
    # to remove numeric digits from string
    list1 = ["".join(filter(lambda x: not x.isdigit(), w)) for w in list1]

    return list1


for index, row in df.iterrows():
    row['proceedTitle'] = Preproceed(row['Title'])
    row['proceedBody'] = Preproceed(row['Body'])
    print(index)

df['proceedTitleBody'] = df['proceedTitle'].str.cat(df['proceedBody'].astype(str), sep=' ')
df.to_csv(PATH_proceedPERSICA, sep='\t', index=False)
