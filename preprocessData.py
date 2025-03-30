from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re, string, copy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import time
startTime = time.time()

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def tokenization(text):
    tokens = re.split(' ', text)
    return tokens


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output


def lemmatize_sentence(text):
    lemmas = m.lemmatize(text)
    return "".join(lemmas).strip()

mode = input("enter the sheet name 1 - rub(1556), 2 - Validation: ")


if mode == "1":
    data = pd.read_excel("..//hhRUtrainingData.xlsx", sheet_name="rub(1556)", usecols=['description', "parsedData.items.salary.from", "parsedData.items.salary.to"])
    columns_to_include = ["testing", "Column1.salary.from", "Column1.salary.to"]
    nameToSave = "testingDataSet.xlsx"

elif mode == "2":
    data = pd.read_excel("..//hhRUtrainingData.xlsx", sheet_name="validation", usecols=['description', "parsedData.items.salary.from", "parsedData.items.salary.to"])
    columns_to_include = ["testing", "parsedData.items.salary.from", "parsedData.items.salary.to"]
    nameToSave = "ValidationTest.xlsx"

stopwords = set(stopwords.words('russian') + ["привет", "мы", "—", "it", "аккредитованная", "компания"])
tfidf_vectorizer = TfidfVectorizer(lowercase=False)

data["punctuation_free"] = data["description"].apply(lambda x: remove_punctuation(x))
data["tokenized"] = data["punctuation_free"].apply(lambda x: tokenization(x))

data['no_stopwords'] = data['tokenized'].apply(lambda x: remove_stopwords(x))

m = Mystem()
data['lemmatizedPyMystem'] = data['punctuation_free'].apply(lambda x: lemmatize_sentence(x))
data["tokenizedLemPyMystem"] = data["lemmatizedPyMystem"].apply(lambda x: tokenization(x))
data['no_stopwordsPyMystem'] = data['tokenizedLemPyMystem'].apply(lambda x: remove_stopwords(x))
data['testing'] = data['no_stopwordsPyMystem'].apply(lambda x: " ".join(x))

print("execution time: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - startTime)))
data[columns_to_include].to_excel(nameToSave, sheet_name="test_preprocessing")