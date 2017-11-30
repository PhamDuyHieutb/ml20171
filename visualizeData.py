from os import listdir
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


# nhan lop
classes = []


def read_file(file_path):
    """
    Read file from disk
    file_path
    """
    file = open(file_path, 'r')
    try:
        text = file.read()
    except UnicodeDecodeError:
        print("fail open file: " + file_path)
        text = ''
    file.close()
    return text

def plot_confusion_matrix(cm,classes,normalize=False,cmap = plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    index = []



    plt.imshow(cm,interpolation= 'nearest',cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 90)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# path data
TRAIN = "preprocessv2/trainprocessed"
TEST = "preprocessv2/testprocessed"

def main():
    y_test = []
    for line in read_file("preprocessv2/datatestsvm").split("\n"):
         y_test.append(line.split(" ")[0])

    y_pred = []
    label_predic_path = '/home/hadoop/PycharmProjects/libsvm-3.22/test/resultofficialv8'
    for line in read_file(label_predic_path).split("\n"):
        y_pred.append(line)

    for sub_folder in listdir(TEST):
        classes.append(sub_folder)

    cnf_matrix = confusion_matrix(y_test,y_pred[:-1])
    # np.set_printoptions(precision= -1)

    plt.figure(figsize=(8, 8))
    plot_confusion_matrix(cnf_matrix,classes=classes,normalize= True)
    plt.show()

if __name__ == '__main__':
    main()







'''
exampletest, labeltest = getInput(TEST)
inputtest = list(zip(exampletest, labeltest))
shuffle(inputtest)
input_train = ([a for a,b in input_train],[b for a,b in input_train])
inputtest = ([a for a,b in inputtest],[b for a,b in inputtest])
print(len(inputtest))
size = len(inputtest[0])
'''
# train = ([input[0][_] for _ in range(size) if _ < 1800],[input[1][_] for _ in range(size) if _ < 1800])
# inputtestsub = ([inputtest[0][_] for _ in range(size) if _ < 3000],[inputtest[1][_] for _ in range(size) if _ < 3000])

'''
def TrainMultiNB(x_train,y_train,alpha):
    mnNB = MultinomialNB(alpha= alpha)
    predicter = mnNB.fit(x_train, y_train)
    return predicter
def mainNB():
    # model = Train_SVM(train[0],train[1])
    ## best param for model
    max_acc = 0
    best_pram = 0
    for i in range(1,11):
        model = TrainMultiNB(input_train[0], input_train[1],i*0.1)
        predicter = predict(inputtest[0],inputtest[1],model)
        print(str(predicter) + "- alpha "+ str(i*0.1))
        if max_acc ==0:
            max_acc = predicter
            best_pram = i * 0.1
        elif predicter > max_acc:
            max_acc = predicter
            best_pram = i*0.1

    print("best alpha = " + str(best_pram))

    model = TrainMultiNB(input_train[0], input_train[1],best_pram)

    confusionMatrix = ConfusionMatrix(inputtest[0],inputtest[1],model)
    print(confusionMatrix)

    # visualize

    plt.title('confusion matrix NB')
    plt.ylabel("True label")
    plt.xlabel("Predict label")
    plt.tight_layout()
    plt.imshow(confusionMatrix)
    plt.show()


def predict(x_test,y_test, predicter):
    y_predict = predicter.predict(x_test)
    print("f1 score")
    print(f1_score(y_test, y_predict, average="macro"))
    print("accurancy")
    accurance = accuracy_score(y_test, y_predict)
    print(accurance)
    print("confusion matrix")
    print(confusion_matrix(y_test, y_predict))
    return accurance

def ConfusionMatrix(x_test,y_test, predicter):
    y_predict = predicter.predict(x_test)
    return confusion_matrix(y_test, y_predict)

def Train_SVM(x_train,y_train,C):
    lsvm = LinearSVC(C= C)
    predicter = lsvm.fit(x_train,y_train)
    return predicter

def Train_SVC(x_train,y_train):
    lsvm = SVC(kernel= "rbf",gamma= 1)
    predicter = lsvm.fit(x_train,y_train)
    return predicter
'''

