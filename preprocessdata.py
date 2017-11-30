import os
from os.path import join
import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from random import shuffle
from DataProcessor import DataProcessor
from os import listdir



def normalize(file):
    """
    loai bo nhung ki tu khong can thiet
    """
    text = file
    match = {
        ' currency ': '(\€|\¥|\£|\$)\d+([\.\,]\d+)*',
        ' email ': '[^\r\n @]+@[^ ]+',
        ' url ': '(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))',
        ' number ': '\d+[\.\,]*\d*',
        '*': '(\'s|\'ll|n\'t|\'re|\'d|\'ve)',
        ' ': '[^a-zA-Z]'
    }
    for key in match:
        text = re.sub(match[key], key, text)
    return text


def ensure_path(path):
    """
    Ensure path exists for write file
    path
    """
    subs = path.split('/')
    full_fill = '.'
    for name in subs[:-1]:
        full_fill += '/{name}'.format(name = name)
        if not os.path.exists(full_fill):
            os.makedirs(full_fill)
    full_fill += str('/'+ subs[-1])
    return full_fill


def load_stop_word(lemmatizer, stemmer):
    """
    Load stop words file
    """
    text = read_file('stopwords.txt')
    words = text.split('\n')
    # words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words]
    # write_file("dictionary2",', '.join(words))
    return words


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
        text =''
    file.close()
    return text


def write_file(file_path, data):
    """
    Write file to disk
    file_path
    data
    """
    file = open(ensure_path(file_path), 'w')
    file.write(data)
    file.close()


def wordenize(lemmatizer, stemmer, text, stop_words):
    """
    Split text into words and stem them
    stemmer -- stemmer object
    text    -- text to wordenize
    """
    text = normalize(text)
    words = nltk.word_tokenize(text)
    newwords = []
    for word in words:
        if word not in stop_words:
            newword = stemmer.stem(lemmatizer.lemmatize(word.lower()))
            newwords.append(newword)
    return newwords


# remove stopword and stem, lemmatizer
def process_data(lemmatizer, stemmer, stop_words,raw_traindata,traindata_restop):
    # read train folder
    train_folders = os.listdir(raw_traindata)
    for folder_name in train_folders:

        # for each folder: list file
        train_files = os.listdir(raw_traindata +'/{folder_name}'.format(folder_name = folder_name))
        print('Working: {folder_name}'.format(folder_name = folder_name))

        for file in train_files:
            # for each file: read file
            text = read_file(raw_traindata + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            if text =='':
                print("detect empty file: " + file +" =>  to the next file  ")
                continue
                # then wordenize it to array of words
            words = wordenize(lemmatizer, stemmer, text, stop_words)

            # write out
            write_file(traindata_restop + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(words))


# caculate tf-idf
def tf_idf(word, current_doc, documents):
    current_count = 0
    current_len = len(current_doc)
    for w in current_doc:
        if word == w:
            current_count += 1
    tf = current_count / current_len

    all_count = 0
    all_len = len(documents)
    for d in documents:
        if word in d:
            all_count += 1
    if all_count == 0:
        all_count = 1
    idf = math.log(all_len / all_count, 10)

    return tf * idf


# convert data to format data for svm
def convertdata(input):
    alldata = []
    for a, b in input:
        data = str(b)
        index = -1
        for tf in a:
            index +=1
            if tf != 0:
                feature = str(index) +":" + str(tf)
                data = data +" "+ feature
        if len(data.split(" ")) > 2:
            alldata.append(data)
        write_file("datatrainsvm", "\n".join(alldata))


#make dict and process train data

def make_dictionary(traindata_restop,final_traindata,path_dict):
    dictionary = []
    # Read words from file
    train_folders = os.listdir(traindata_restop)
    words = []
    for folder_name in train_folders:
        # for each folder: list file
        train_files = os.listdir(traindata_restop + '/{folder_name}'.format(folder_name = folder_name))
        print('Working: {folder_name}'.format(folder_name = folder_name))
        line = []

        for file in train_files:
            # for each file: read file
            text = read_file(traindata_restop + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))

            line += text.split(', ');

        words.append(line);

    for i in range(len(words)):
        line = list(set(words[i]))
        freq = {}
        for word in line:
            freq[word] = 0   # gan tan so ban dau cho cac tu bang 0
        for j in range(len(words[i])):
            freq[words[i][j]] += 1     # tinh tan so cua cac tu
        line = [w for w in line if (freq[w] > 4 and freq[w] < 4000)] ## loai bo cac tu co tan so qua cao hoac qua thap
        print('=====================> ', i, ' ============ ', len(line))   ## do dai cua document hien tai
        j = 0
        new_line = []
        for word in line:    # tinh tf-idf cho cac tu trong 1 document
            # print (j, ' - ', word)
            value = tf_idf(word, words[i], words)
            if value >= 2e-05:
                new_line.append(word)
            j += 1
        dictionary += new_line

    print('dictionary before: ', len(dictionary))
    # remove word which appears in documents > 3000 or < 3
    listwordremove = []
    for word in dictionary:
        count = 0
        for folder_name in train_folders:
            train_files = os.listdir(traindata_restop + '/{folder_name}'.format(folder_name=folder_name))
            for file in train_files:
                text = read_file(traindata_restop + '/{folder_name}/{file}'.format(folder_name=folder_name,
                                                                                      file=file))
                for w in text.split(", "):      # if word in dict which appears in file => to the next file
                    if word == w:
                        count+=1
                        break

        if count > 3000 or count < 4:    # remove word which appears in documents > 3000 or < 4
            listwordremove.append(word)
            print("word be removed:" + word+ "-" + str(count) +"\n")

    for w in listwordremove:
        dictionary.remove(w)



    # write out
    write_file(path_dict, ', '.join(dictionary))

    print('dictionary after remove: ', len(dictionary))
    print('train_folder: ', len(train_folders))

    # ghi data filtered theo dict vao processed
    for i in range(len(train_folders)):
        print('i: ', i)
        folder_name = train_folders[i]
        train_files = os.listdir(traindata_restop + '/{folder_name}'.format(folder_name = folder_name))
        for file in train_files:
            text = read_file(traindata_restop + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            arr_text = text.split(', ');
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            print('Write: {folder_name}/{file}'.format(folder_name = folder_name, file =file))
            write_file( final_traindata + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(elements_in_both_lists))


# process data test by dict which made for train data

def filterTestData(testdata_restop,final_testdata,path_dict):
    dictionary = read_file(path_dict).split(", ")  # load dict to filter test data
    print('dictionary : ', len(dictionary))

    # ghi data filtered theo dict vao processed
    train_folders = os.listdir(testdata_restop)
    for i in range(len(train_folders)):
        print('i: ', i)
        folder_name = train_folders[i]
        train_files = os.listdir(testdata_restop + '/{folder_name}'.format(folder_name = folder_name))
        for file in train_files:
            text = read_file(testdata_restop + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            arr_text = text.split(', ');
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            print('Write: {folder_name}/{file}'.format(folder_name = folder_name, file =file))
            write_file(final_testdata + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(elements_in_both_lists))



'''
this code : convert data preprocessed to svm format data 
and visualize result
'''

dataprocessor = DataProcessor()

def getInput(path, train=False):
    #   GET DATA    #

    corpus = []
    label = []
    counter = 0

    for sub_folder in listdir(path):
        pathsub = join(path, sub_folder)
        for file_name in listdir(pathsub):
            content = open(join(pathsub, file_name)).read().rstrip()
            label.append(counter)
            doc = ' '.join(content.split(", "))
            corpus.append(doc)
        counter += 1

        #
    if train:
        tfidf = dataprocessor.fit(corpus)
    else:
        tfidf = dataprocessor.transform(corpus)

    return tfidf, label

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

def write_file(file_path, data):
    """
    Write file to disk
    file_path
    data
    """
    file = open(file_path, 'w')
    file.write(data)
    file.close()

# convert data to format for svm
def convertdata(input, path):
    alldata = []
    for a, b in input:
        data = str(b)
        index = -1
        for tf in a:
            index += 1
            if tf != 0:
                feature = str(index) + ":" + str(tf)
                data = data + " " + feature
        if len(data.split(" ")) > 3:
            alldata.append(data)
        write_file(path, "\n".join(alldata))

# INPUT ARRAY
def ConvertAllDataToSvm(path_traindata,path_testdata):
    inputtrain, labeltrain = getInput(path_traindata, train=True)
    input_train = list(zip(inputtrain, labeltrain))
    shuffle(input_train)

    exampletest, labeltest = getInput(path_testdata)
    inputtest = list(zip(exampletest, labeltest))
    shuffle(inputtest)

    convertdata(input_train, "preprocessv2/datatrainsvm")
    convertdata(inputtest, "preprocessv2/datatestsvm")


# path folder raw data
FOLDER_RAW_TRAINDATA = "20news-bydate/20news-bydate-train"
FOLDER_RAW_TESTDATA = "20news-bydate/20news-bydate-test"

#data after remove stopword
FOLDER_TRAINDATA_RESTOP = "preprocessv2/train"
FOLDER_TESTDATA_RESTOP  = "preprocessv2/test"

# data after filter by dictionary (final)
FOLDER_FINAL_TRAINDATA   = "preprocessv2/trainprocessed"
FOLDER_FINAL_TESTDATA   = "preprocessv2/testprocessed"
# path dictionary
PATH_DICT      = "preprocessv2/dictionary"



def main():
    # """ MAIN FUNCTION """

    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = load_stop_word(lemmatizer, stemmer)
    process_data(lemmatizer, stemmer, stop_words,FOLDER_RAW_TRAINDATA,FOLDER_TRAINDATA_RESTOP)
    process_data(lemmatizer, stemmer, stop_words, FOLDER_RAW_TESTDATA, FOLDER_TESTDATA_RESTOP)

    # make_dictionary(stemmer) and process train,test data
    make_dictionary(FOLDER_TRAINDATA_RESTOP,FOLDER_FINAL_TRAINDATA,PATH_DICT)
    filterTestData(FOLDER_TESTDATA_RESTOP,FOLDER_FINAL_TESTDATA,PATH_DICT)

if __name__ == '__main__':
    main()