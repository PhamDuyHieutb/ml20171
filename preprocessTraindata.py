import os
from os.path import join
import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def normalize(file):
    """
    loai bo nhung ki tu khong can th
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

# path folder data training after processed
FOLDER_RAWDATA = "20news-bydate/20news-bydate-train"
FOLDER_OUTPUT  = "preprocessv2/train"
PATH_DICT      = "preprocessv2/dictionary"
FOLDER_FINAL   = "preprocessv2/trainprocessed"
# FOLDER_RAWDATA = "testsample/trainfolder/20news_trainraw"
# FOLDER_OUTPUT  = "testsample/trainfolder/preprocesstrain"
# PATH_DICT      = "testsample/trainfolder/dic2"
# FOLDER_FINAL   = "testsample/trainfolder/finaltrain"
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

def make_dictionary():
    dictionary = []
    # Read words from file
    train_folders = os.listdir(FOLDER_OUTPUT)
    words = []
    for folder_name in train_folders:
        # for each folder: list file
        train_files = os.listdir(FOLDER_OUTPUT + '/{folder_name}'.format(folder_name = folder_name))
        print('Working: {folder_name}'.format(folder_name = folder_name))
        line = []

        for file in train_files:
            # for each file: read file
            text = read_file(FOLDER_OUTPUT + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))

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
            train_files = os.listdir(FOLDER_OUTPUT + '/{folder_name}'.format(folder_name=folder_name))
            for file in train_files:
                text = read_file(FOLDER_OUTPUT + '/{folder_name}/{file}'.format(folder_name=folder_name,
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
    write_file(PATH_DICT, ', '.join(dictionary))

    print('dictionary after remove: ', len(dictionary))
    print('train_folder: ', len(train_folders))

    # ghi data filtered theo dict vao processed
    for i in range(len(train_folders)):
        print('i: ', i)
        folder_name = train_folders[i]
        train_files = os.listdir(FOLDER_OUTPUT + '/{folder_name}'.format(folder_name = folder_name))
        for file in train_files:
            text = read_file(FOLDER_OUTPUT + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file))
            arr_text = text.split(', ');
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            print('Write: {folder_name}/{file}'.format(folder_name = folder_name, file =file))
            write_file(  FOLDER_FINAL + '/{folder_name}/{file}'.format(folder_name = folder_name, file =file), ', '.join(elements_in_both_lists))


def main():
    # """ MAIN FUNCTION """

    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = load_stop_word(lemmatizer, stemmer)
    process_data(lemmatizer, stemmer, stop_words)

    # make_dictionary(stemmer)
    make_dictionary()


if __name__ == '__main__':
    main()