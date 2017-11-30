import os
from os.path import join
import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

'''
this code to preprocess test data with dictionary which made by train data
'''


def normalize(file):
    """
    Normalize text: replace special paterns by word
    file
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
    Load stop words file and stem them
    stemmer
    """
    text = read_file('stopwords.txt')
    words = text.split('\n')
    # words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words]
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
    words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words if word not in stop_words]
    return words

# path folder data training after processed

# FOLDER_RAWDATA = "testsample/testfolder/rawtest"
# FOLDER_OUTPUT  = "testsample/testfolder/preprocess"
# PATH_DICT      = "testsample/trainfolder/dic2"
FOLDER_RAWDATA = "20news-bydate/20news-bydate-test"
FOLDER_OUTPUT  = "preprocessv2/test"
PATH_DICT      = "preprocessv2/dictionary"
FOLDER_FINAL   = "preprocessv2/testprocessed"

def process_data(lemmatizer, stemmer, stop_words):
    # read train folder
    train_folders = os.listdir(FOLDER_RAWDATA)
    for folder_name in train_folders:

        # for each folder: list file
        train_files = os.listdir( FOLDER_RAWDATA +'/{folder_name}'.format(folder_name = folder_name))
        print('Working: {folder_name}'.format(folder_name = folder_name))

        for file in train_files:
            # for each file: read file
            text = read_file(FOLDER_RAWDATA + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            if text =='':
                print("detect empty file: " + file +" =>  to the next file  ")
                continue
                # then wordenize it to array of words
            words = wordenize(lemmatizer, stemmer, text, stop_words)

            # write out
            write_file(FOLDER_OUTPUT + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(words))


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


def filterData():
    dictionary = read_file(PATH_DICT).split(", ")  # load dict to filter test data
    print('dictionary : ', len(dictionary))

    # ghi data filtered theo dict vao processed
    train_folders = os.listdir(FOLDER_OUTPUT)
    for i in range(len(train_folders)):
        print('i: ', i)
        folder_name = train_folders[i]
        train_files = os.listdir(FOLDER_OUTPUT + '/{folder_name}'.format(folder_name = folder_name))
        for file in train_files:
            text = read_file(FOLDER_OUTPUT + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            arr_text = text.split(', ');
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            print('Write: {folder_name}/{file}'.format(folder_name = folder_name, file =file))
            write_file( FOLDER_FINAL + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(elements_in_both_lists))

def main():

    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = load_stop_word(lemmatizer, stemmer)
    process_data(lemmatizer, stemmer, stop_words)

    # make_dictionary(stemmer)
    filterData()


if __name__ == '__main__':
    main()