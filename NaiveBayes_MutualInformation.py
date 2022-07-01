from __future__ import unicode_literals
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from skfeature.function.information_theoretical_based import MRMR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import product
import codecs
import string
import time


CV = 10
REPEAT = 10
PATH_PROCEEDPERSICA = 'proceedPERSICA.txt'
PROCEED_DATA_COL = 'proceedTitleBody'
# PROCEED_DATA_COL = 'proceedBody'
# PROCEED_DATA_COL = 'proceedTitle'

PATH_MINaiveBayes_Results = 'MI_NaiveBayes'+PROCEED_DATA_COL+'_Results.txt'

# perform evaluation on classification task
PERCENT_FEATURE = 0.03  # number of selected features
max_bow_features = None
NUM_CLASS = 11

def text_process(mess):
    return [word for word in mess.split()]


def myplotconfusion(cm, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None, display_labels=[]):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        n_classes = cm.shape[0]
        im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        text_ = None

        cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

        if include_values:
            text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
        return fig, ax


def plot_confusion(cm, title1):
    np.set_printoptions(precision=4)
    fig, ax = myplotconfusion(cm, cmap=plt.cm.Blues)
    ax.set_title(title1)
    fig.set_size_inches(10, 10)
    fig.savefig(str(title1) + '.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


txt = pd.read_csv(PATH_PROCEEDPERSICA, sep='\t', encoding='UTF-8')

all_txt = txt[PROCEED_DATA_COL]
all_label = txt['Category2']

le = preprocessing.LabelEncoder()
transform_all_label = le.fit_transform(all_label)

rskf = RepeatedStratifiedKFold(n_splits=CV, n_repeats=REPEAT, random_state=36851234)
MINB_Accs = np.zeros(CV*REPEAT)
MINB_p = np.zeros(CV*REPEAT)
MINB_r = np.zeros(CV*REPEAT)
MINB_f = np.zeros(CV*REPEAT)

MINB_perClass_p = np.zeros((CV*REPEAT, NUM_CLASS))
MINB_perClass_r = np.zeros((CV*REPEAT, NUM_CLASS))
MINB_perClass_f = np.zeros((CV*REPEAT, NUM_CLASS))

MINB_maxAcc = 0
cvid=0

start_time = time.time()
for train_index, test_index in rskf.split(all_txt, all_label):
    txt_train, txt_test = all_txt[train_index], all_txt[test_index]
    label_train, label_test = all_label[train_index], all_label[test_index]

    bow_transformer = CountVectorizer(analyzer=text_process,  max_features=max_bow_features) # max_df': (0.5, 0.75, 1.0) , max_df=0.95, min_df=2,
    bow_transformer.fit(txt_train)
    messages_bow = bow_transformer.transform(txt_train)
    # print('Shape of Sparse Matrix: ', messages_bow.shape)

    tfidf_transformer = TfidfTransformer(norm='l2').fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    max_bow_features = messages_tfidf.shape[1]
    NUM_FEATURE = int(max_bow_features*PERCENT_FEATURE)
    # print("messages tfidf.shape: ")
    # print(messages_tfidf.shape)

    # mi = mutual_info_classif(messages_tfidf.toarray(), label_train, discrete_features=False)
    # idx = mi.argsort()[-NUM_FEATURE:][::-1]
    # obtain the dataset on the selected features
    # selfeature_messages_tfidf = messages_tfidf[:, idx[0:NUM_FEATURE]]

    sel_mutual = SelectKBest(mutual_info_classif, k=NUM_FEATURE)
    selfeature_messages_tfidf = sel_mutual.fit_transform(messages_tfidf, label_train)
    # print(sel_mutual.get_support())

    MultinomialMINB__model = MultinomialNB(fit_prior=True).fit(selfeature_messages_tfidf, label_train)


    test_messages_bow = bow_transformer.transform(txt_test)
    # print('Shape of test_messages_bow Sparse Matrix: ', test_messages_bow.shape)

    test_messages_tfidf = tfidf_transformer.transform(test_messages_bow)
    selfeature_test_messages_tfidf = sel_mutual.transform(test_messages_tfidf)
    # selfeature_test_messages_tfidf = test_messages_tfidf[:, idx[0:NUM_FEATURE]]
    # print("messages tfidf.shape: ")
    # print(test_messages_tfidf.shape)


    predictions = MultinomialMINB__model.predict(selfeature_test_messages_tfidf)
    # print(classification_report(label_test, predictions))

    # 'macro': Calculate metrics for each label, and find their unweighted mean.This does not take label imbalance into account.
    (precision, recall, fscore, support) = precision_recall_fscore_support(label_test, predictions, average='macro')
    # print(precision, recall, fscore, support)
    # cm = confusion_matrix(label_test, predictions, normalize='all')
    # print(cm)

    # If None, the scores for each class are returned
    (MINB_perClass_p[cvid, :], MINB_perClass_r[cvid, :], MINB_perClass_f[cvid, :], support) = precision_recall_fscore_support(
        label_test, predictions, average=None)

    acc = accuracy_score(label_test, predictions)
    MINB_Accs[cvid] = acc
    (MINB_p[cvid], MINB_r[cvid], MINB_f[cvid]) = (precision, recall, fscore)

    cvid += 1
    if acc > MINB_maxAcc:
        MINB_maxAcc = acc
        (MINB_maxp, MINB_maxr, MINB_maxf) = (precision, recall, fscore)
        MINB_bestConfusion_normalize = confusion_matrix(label_test, predictions, normalize='all')
        MINB_bestConfusion_none = confusion_matrix(label_test, predictions, normalize=None)

print("MI & Naive Bayes: Avg MI & Naive Bayes   Accuracy", np.average(MINB_Accs), "+- ", np.std(MINB_Accs))
print("MI & Naive Bayes: Avg MI & Naive Bayes   precision", np.average(MINB_p), "+- ", np.std(MINB_p))
print("MI & Naive Bayes: Avg MI & Naive Bayes   recall", np.average(MINB_r), "+- ", np.std(MINB_r))
print("MI & Naive Bayes: Avg MI & Naive Bayes   fscore", np.average(MINB_f), "+- ", np.std(MINB_f))
print("MI & Naive Bayes: Best Accuracy:", MINB_maxAcc, "precision: ", MINB_maxp, "recall: ", MINB_maxr, "fscore: ", MINB_maxf)
print("MI & Naive Bayes: Best Confusion", MINB_bestConfusion_normalize)
plot_confusion(MINB_bestConfusion_normalize, "MINaive Bayes Normalized confusion matrix")
plot_confusion(MINB_bestConfusion_none, "MINaive Bayes Confusion matrix, without normalization")
# plot_confusion(MINB_bestModel, MINB_y_test, MINB_bestprediction, [])
print("--- %s seconds ---" % (time.time() - start_time))


with codecs.open(PATH_MINaiveBayes_Results, 'w', encoding='utf-8') as f:
    f.write(''.join(["NUM OF BOW FEATUREs : \t ", str(max_bow_features), "\n"]))
    f.write(''.join(['PERCENT: \t', str(PERCENT_FEATURE), " \t NUM OF SELECTED FEATUREs : \t ", str(NUM_FEATURE), "\n"]))
    f.write("\n")
    f.write("\t")
    A = le.inverse_transform(transform_all_label)
    uA = pd.unique(A)

    for index in range(uA.size):
        f.write(uA[index]+"\t")

    f.write("\n")
    f.write("precision \t")
    preci = np.average(MINB_perClass_p, axis=0)
    for index in range(uA.size):
        f.write(str(preci[index])+"\t")

    f.write("\n")
    f.write("recall \t")
    reca = np.average(MINB_perClass_r, axis=0)
    for index in range(uA.size):
        f.write(str(reca[index])+"\t")

    f.write("\n")
    f.write("fscore \t")
    fsc = np.average(MINB_perClass_f, axis=0)
    for index in range(uA.size):
        f.write(str(fsc[index])+"\t")


    f.write("\n\n")

    f.write("\t Average \t std \n")
    f.write(''.join(
        ["MI & Naive Bayes: Avg MI & Naive Bayes   Accuracy \t", str(np.average(MINB_Accs)), "\t", str(np.std(MINB_Accs)), '\n']))
    f.write(''.join(
        ["MI & Naive Bayes: Avg MI & Naive Bayes   precision \t", str(np.average(MINB_p)), "\t", str(np.std(MINB_p)), '\n']))
    f.write(''.join(
        ["MI & Naive Bayes: Avg MI & Naive Bayes   recall \t", str(np.average(MINB_r)), "\t", str(np.std(MINB_r)), '\n']))
    f.write(''.join(
        ["MI & Naive Bayes: Avg MI & Naive Bayes   fscore \t", str(np.average(MINB_f)), "\t", str(np.std(MINB_f)), '\n']))
    f.write("\n")
    f.write(''.join(
        ["MI & Naive Bayes: Best Accuracy: \t", str(MINB_maxAcc), '\n']))
    f.write(''.join(
        ["MI & Naive Bayes:  precision: \t", np.str(MINB_maxp), '\n']))
    f.write(''.join(
        ["MI & Naive Bayes:  recall: \t", np.str(MINB_maxr),  '\n']))
    f.write(''.join(
        ["MI & Naive Bayes:  fscore: \t", np.str(MINB_maxf), '\n']))


