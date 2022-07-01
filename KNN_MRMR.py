from __future__ import unicode_literals
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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


CV = 3
REPEAT = 1
PATH_PROCEEDPERSICA = 'proceedPERSICAsmall.txt'
# PROCEED_DATA_COL = 'proceedTitleBody'
# PROCEED_DATA_COL = 'proceedBody'
PROCEED_DATA_COL = 'proceedTitle'
K = 1
PATH_MRMRKNN_Results = 'KNN_MRMR_K'+str(K)+'_'+PROCEED_DATA_COL+'_Results.txt'

# perform evaluation on classification task
PERCENT_FEATURE = 0.1  # number of selected features

max_bow_features = 50 # None
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
MRMRKNN_Accs = np.zeros(CV*REPEAT)
MRMRKNN_p = np.zeros(CV*REPEAT)
MRMRKNN_r = np.zeros(CV*REPEAT)
MRMRKNN_f = np.zeros(CV*REPEAT)

MRMRKNN_perClass_p = np.zeros((CV*REPEAT, NUM_CLASS))
MRMRKNN_perClass_r = np.zeros((CV*REPEAT, NUM_CLASS))
MRMRKNN_perClass_f = np.zeros((CV*REPEAT, NUM_CLASS))

MRMRKNN_maxAcc = 0
cvid=0

for train_index, test_index in rskf.split(all_txt, all_label):
    txt_train, txt_test = all_txt[train_index], all_txt[test_index]
    label_train, label_test = all_label[train_index], all_label[test_index]

    bow_transformer = CountVectorizer(analyzer=text_process, max_df=1, max_features=max_bow_features)
    bow_transformer.fit(txt_train)
    messages_bow = bow_transformer.transform(txt_train)
    # print('Shape of Sparse Matrix: ', messages_bow.shape)

    tfidf_transformer = TfidfTransformer(norm='l2').fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    max_bow_features = messages_tfidf.shape[1]
    NUM_FEATURE = int(max_bow_features*PERCENT_FEATURE)
    # print("messages tfidf.shape: ")
    # print(messages_tfidf.shape)

    # obtain the index of each feature on the training set
    idx = MRMR.mrmr(messages_tfidf.toarray(), label_train, n_selected_features=NUM_FEATURE)
    # obtain the dataset on the selected features
    selfeature_messages_tfidf = messages_tfidf[:, idx[0:NUM_FEATURE]]

    KNN_model = KNeighborsClassifier(n_neighbors=K).fit(selfeature_messages_tfidf, label_train)


    test_messages_bow = bow_transformer.transform(txt_test)
    # print('Shape of test_messages_bow Sparse Matrix: ', test_messages_bow.shape)

    test_messages_tfidf = tfidf_transformer.transform(test_messages_bow)
    selfeature_test_messages_tfidf = test_messages_tfidf[:, idx[0:NUM_FEATURE]]
    # print("messages tfidf.shape: ")
    # print(test_messages_tfidf.shape)

    predictions = KNN_model.predict(selfeature_test_messages_tfidf)
    # print(classification_report(label_test, predictions))

    # 'macro': Calculate metrics for each label, and find their unweighted mean.This does not take label imbalance into account.
    (precision, recall, fscore, support) = precision_recall_fscore_support(label_test, predictions, average='macro')
    # print(precision, recall, fscore, support)
    # cm = confusion_matrix(label_test, predictions, normalize='all')
    # print(cm)

    # If None, the scores for each class are returned
    (MRMRKNN_perClass_p[cvid, :], MRMRKNN_perClass_r[cvid, :], MRMRKNN_perClass_f[cvid, :], support) = precision_recall_fscore_support(
        label_test, predictions, average=None)

    acc = accuracy_score(label_test, predictions)
    MRMRKNN_Accs[cvid] = acc
    (MRMRKNN_p[cvid], MRMRKNN_r[cvid], MRMRKNN_f[cvid]) = (precision, recall, fscore)

    cvid += 1
    if acc > MRMRKNN_maxAcc:
        MRMRKNN_maxAcc = acc
        (MRMRKNN_maxp, MRMRKNN_maxr, MRMRKNN_maxf) = (precision, recall, fscore)
        MRMRKNN_bestConfusion_normalize = confusion_matrix(label_test, predictions, normalize='all')
        MRMRKNN_bestConfusion_none = confusion_matrix(label_test, predictions, normalize=None)

print("MRMR & KNN: Avg MRMR & KNN   Accuracy", np.average(MRMRKNN_Accs), "+- ", np.std(MRMRKNN_Accs))
print("MRMR & KNN: Avg MRMR & KNN   precision", np.average(MRMRKNN_p), "+- ", np.std(MRMRKNN_p))
print("MRMR & KNN: Avg MRMR & KNN   recall", np.average(MRMRKNN_r), "+- ", np.std(MRMRKNN_r))
print("MRMR & KNN: Avg MRMR & KNN   fscore", np.average(MRMRKNN_f), "+- ", np.std(MRMRKNN_f))
print("MRMR & KNN: Best Accuracy:", MRMRKNN_maxAcc, "precision: ", MRMRKNN_maxp, "recall: ", MRMRKNN_maxr, "fscore: ", MRMRKNN_maxf)
print("MRMR & KNN: Best Confusion", MRMRKNN_bestConfusion_normalize)
plot_confusion(MRMRKNN_bestConfusion_normalize, "MRMRKNN K"+str(K)+"  Normalized confusion matrix")
plot_confusion(MRMRKNN_bestConfusion_none, "MRMRKNN K"+str(K)+"  Confusion matrix, without normalization")
# plot_confusion(MRMRKNN_bestModel, MRMRKNN_y_test, MRMRKNN_bestprediction, [])

with codecs.open(PATH_MRMRKNN_Results, 'w', encoding='utf-8') as f:
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
    preci = np.average(MRMRKNN_perClass_p, axis=0)
    for index in range(uA.size):
        f.write(str(preci[index])+"\t")

    f.write("\n")
    f.write("recall \t")
    reca = np.average(MRMRKNN_perClass_r, axis=0)
    for index in range(uA.size):
        f.write(str(reca[index])+"\t")

    f.write("\n")
    f.write("fscore \t")
    fsc = np.average(MRMRKNN_perClass_f, axis=0)
    for index in range(uA.size):
        f.write(str(fsc[index])+"\t")


    f.write("\n\n")

    f.write("\t Average \t std \n")
    f.write(''.join(
        ["MRMR & KNN: Avg MRMR & KNN   Accuracy \t", str(np.average(MRMRKNN_Accs)), "\t", str(np.std(MRMRKNN_Accs)), '\n']))
    f.write(''.join(
        ["MRMR & KNN: Avg MRMR & KNN   precision \t", str(np.average(MRMRKNN_p)), "\t", str(np.std(MRMRKNN_p)), '\n']))
    f.write(''.join(
        ["MRMR & KNN: Avg MRMR & KNN   recall \t", str(np.average(MRMRKNN_r)), "\t", str(np.std(MRMRKNN_r)), '\n']))
    f.write(''.join(
        ["MRMR & KNN: Avg MRMR & KNN   fscore \t", str(np.average(MRMRKNN_f)), "\t", str(np.std(MRMRKNN_f)), '\n']))
    f.write("\n")
    f.write(''.join(
        ["MRMR & KNN: Best Accuracy: \t", str(MRMRKNN_maxAcc), '\n']))
    f.write(''.join(
        ["MRMR & KNN:  precision: \t", np.str(MRMRKNN_maxp), '\n']))
    f.write(''.join(
        ["MRMR & KNN:  recall: \t", np.str(MRMRKNN_maxr),  '\n']))
    f.write(''.join(
        ["MRMR & KNN:  fscore: \t", np.str(MRMRKNN_maxf), '\n']))



