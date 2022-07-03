from genericpath import isdir
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
import os


class Cleaner:

    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool,
                 perform_accents_removal: bool, perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        # altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = dict(
            zip(list(in_table), list(out_table)))
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        return BeautifulSoup(html_doc, "html.parser").get_text()

    @staticmethod
    def read_stop_words(str_file) -> set:
        set_stop_words = set()
        with open(str_file, encoding='utf-8') as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return term in self.set_stop_words

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        new_term = ""
        for char in list(term):
            if (char in self.accents_translation_table.keys()):
                char = self.accents_translation_table[char]
            new_term += char

        return new_term

    def preprocess_word(self, term: str) -> str or None:
        term = term.lower()
        if (term in self.set_punctuation):
            return None
        if (self.perform_stop_words_removal and term in self.set_stop_words):
            return None
        if (self.perform_accents_removal):
            term = self.remove_accents(term)
        if (self.perform_stemming):
            term = self.word_stem(term)
        return term

    def preprocess_text(self, text: str) -> str or None:
        return self.remove_accents(text.lower())


class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                      language="portuguese",
                      perform_stop_words_removal=True,
                      perform_accents_removal=True,
                      perform_stemming=True)

    def __init__(self, index):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = {}
        list_of_words = word_tokenize(plain_text)
        for word in list_of_words:
            processed_word = self.cleaner.preprocess_word(word)
            if processed_word is None:
                continue
            if processed_word in dic_word_count.keys():
                dic_word_count[processed_word] += 1
            else:
                dic_word_count[processed_word] = 1
        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        plain_text = self.cleaner.html_to_plain_text(text_html)
        dic_word_count = self.text_word_count(plain_text)
        for word, freq in dic_word_count.items():
            self.index.index(word, doc_id, freq)

    def index_text_dir(self, path: str):
        # from tqdm import tqdm
        # for str_sub_dir in tqdm(os.listdir(path)):
        for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"
            if (os.path.isdir(path_sub_dir)):
                self.index_text_dir(path_sub_dir)

            elif (os.path.isfile(path_sub_dir)
                  and str_sub_dir.endswith(".html")):
                with open(path_sub_dir, encoding="utf-8") as f:
                    file_name = os.path.basename(path_sub_dir)
                    doc_id = int(file_name.split('.')[0])
                    text_html = f.read()
                    self.index_text(doc_id, text_html)

        self.index.finish_indexing()
