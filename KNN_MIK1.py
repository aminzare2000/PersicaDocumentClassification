from __future__ import unicode_literals
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
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
K = 9
PATH_MIKNN_Results = 'KNN_MI_K'+str(K)+'_'+PROCEED_DATA_COL+'_Results.txt'

# perform evaluation on classification task
PERCENT_FEATURE = 0.5  # number of selected features
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
MIKNN_Accs = np.zeros(CV*REPEAT)
MIKNN_p = np.zeros(CV*REPEAT)
MIKNN_r = np.zeros(CV*REPEAT)
MIKNN_f = np.zeros(CV*REPEAT)

MIKNN_perClass_p = np.zeros((CV*REPEAT, NUM_CLASS))
MIKNN_perClass_r = np.zeros((CV*REPEAT, NUM_CLASS))
MIKNN_perClass_f = np.zeros((CV*REPEAT, NUM_CLASS))

MIKNN_maxAcc = 0
cvid=0

start_time = time.time()
for train_index, test_index in rskf.split(all_txt, all_label):
    txt_train, txt_test = all_txt[train_index], all_txt[test_index]
    label_train, label_test = all_label[train_index], all_label[test_index]

    bow_transformer = CountVectorizer(analyzer=text_process, max_features=max_bow_features) # max_df': (0.5, 0.75, 1.0) , max_df=0.95, min_df=2,
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

    KNN_model = KNeighborsClassifier(n_neighbors=K).fit(messages_tfidf, label_train)


    test_messages_bow = bow_transformer.transform(txt_test)
    # print('Shape of test_messages_bow Sparse Matrix: ', test_messages_bow.shape)

    test_messages_tfidf = tfidf_transformer.transform(test_messages_bow)
    selfeature_test_messages_tfidf = sel_mutual.transform(test_messages_tfidf)
    # selfeature_test_messages_tfidf = test_messages_tfidf[:, idx[0:NUM_FEATURE]]
    # print("messages tfidf.shape: ")
    # print(test_messages_tfidf.shape)

    predictions = KNN_model.predict(test_messages_tfidf)
    # print(classification_report(label_test, predictions))

    # 'macro': Calculate metrics for each label, and find their unweighted mean.This does not take label imbalance into account.
    (precision, recall, fscore, support) = precision_recall_fscore_support(label_test, predictions, average='macro')
    # print(precision, recall, fscore, support)
    # cm = confusion_matrix(label_test, predictions, normalize='all')
    # print(cm)

    # If None, the scores for each class are returned
    (MIKNN_perClass_p[cvid, :], MIKNN_perClass_r[cvid, :], MIKNN_perClass_f[cvid, :], support) = precision_recall_fscore_support(
        label_test, predictions, average=None)

    acc = accuracy_score(label_test, predictions)
    MIKNN_Accs[cvid] = acc
    (MIKNN_p[cvid], MIKNN_r[cvid], MIKNN_f[cvid]) = (precision, recall, fscore)

    cvid += 1
    if acc > MIKNN_maxAcc:
        MIKNN_maxAcc = acc
        (MIKNN_maxp, MIKNN_maxr, MIKNN_maxf) = (precision, recall, fscore)
        MIKNN_bestConfusion_normalize = confusion_matrix(label_test, predictions, normalize='all')
        MIKNN_bestConfusion_none = confusion_matrix(label_test, predictions, normalize=None)

print("MI & KNN: Avg MI & KNN   Accuracy", np.average(MIKNN_Accs), "+- ", np.std(MIKNN_Accs))
print("MI & KNN: Avg MI & KNN   precision", np.average(MIKNN_p), "+- ", np.std(MIKNN_p))
print("MI & KNN: Avg MI & KNN   recall", np.average(MIKNN_r), "+- ", np.std(MIKNN_r))
print("MI & KNN: Avg MI & KNN   fscore", np.average(MIKNN_f), "+- ", np.std(MIKNN_f))
print("MI & KNN: Best Accuracy:", MIKNN_maxAcc, "precision: ", MIKNN_maxp, "recall: ", MIKNN_maxr, "fscore: ", MIKNN_maxf)
print("MI & KNN: Best Confusion", MIKNN_bestConfusion_normalize)
plot_confusion(MIKNN_bestConfusion_normalize, "MIKNN K"+str(K)+"  Normalized confusion matrix")
plot_confusion(MIKNN_bestConfusion_none, "MIKNN K"+str(K)+" Confusion matrix, without normalization")
# plot_confusion(MIKNN_bestModel, MIKNN_y_test, MIKNN_bestprediction, [])
print("--- %s seconds ---" % (time.time() - start_time))

with codecs.open(PATH_MIKNN_Results, 'w', encoding='utf-8') as f:
    f.write(''.join(["NUM OF BOW FEATUREs : \t ", str(max_bow_features), "\n"]))
    f.write(''.join(['PERCENT: \t', str(PERCENT_FEATURE), " \t NUM OF SELECTED FEATUREs : \t ", str(NUM_FEATURE), "\n"]))
    f.write(''.join(["KNN, K=\t ", str(K), "\n"]))
    f.write("\n")
    f.write("\t")
    A = le.inverse_transform(transform_all_label)
    uA = pd.unique(A)

    for index in range(uA.size):
        f.write(uA[index]+"\t")

    f.write("\n")
    f.write("precision \t")
    preci = np.average(MIKNN_perClass_p, axis=0)
    for index in range(uA.size):
        f.write(str(preci[index])+"\t")

    f.write("\n")
    f.write("recall \t")
    reca = np.average(MIKNN_perClass_r, axis=0)
    for index in range(uA.size):
        f.write(str(reca[index])+"\t")

    f.write("\n")
    f.write("fscore \t")
    fsc = np.average(MIKNN_perClass_f, axis=0)
    for index in range(uA.size):
        f.write(str(fsc[index])+"\t")


    f.write("\n\n")

    f.write("\t Average \t std \n")
    f.write(''.join(
        ["MI & KNN: Avg MI & KNN   Accuracy \t", str(np.average(MIKNN_Accs)), "\t", str(np.std(MIKNN_Accs)), '\n']))
    f.write(''.join(
        ["MI & KNN: Avg MI & KNN   precision \t", str(np.average(MIKNN_p)), "\t", str(np.std(MIKNN_p)), '\n']))
    f.write(''.join(
        ["MI & KNN: Avg MI & KNN   recall \t", str(np.average(MIKNN_r)), "\t", str(np.std(MIKNN_r)), '\n']))
    f.write(''.join(
        ["MI & KNN: Avg MI & KNN   fscore \t", str(np.average(MIKNN_f)), "\t", str(np.std(MIKNN_f)), '\n']))
    f.write("\n")
    f.write(''.join(
        ["MI & KNN: Best Accuracy: \t", str(MIKNN_maxAcc), '\n']))
    f.write(''.join(
        ["MI & KNN:  precision: \t", np.str(MIKNN_maxp), '\n']))
    f.write(''.join(
        ["MI & KNN:  recall: \t", np.str(MIKNN_maxr),  '\n']))
    f.write(''.join(
        ["MI & KNN:  fscore: \t", np.str(MIKNN_maxf), '\n']))



