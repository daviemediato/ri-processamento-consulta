import enum
from typing import List
from abc import abstractmethod
from typing import List, Set, Mapping
from index.structure import TermOccurrence
import math
from enum import Enum


class IndexPreComputedVals():

    def __init__(self, index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        self.doc_count = self.index.document_count
        self.calculate_norm()

    def calculate_norm(self):
        for document_id in self.index.set_documents:
            norm_list = []
            norm = 0

            for word in self.index.vocabulary:
                for term in self.index.get_occurrence_list(word):
                    if term.doc_id == document_id:
                        tf_idf = VectorRankingModel.tf_idf(
                            self.doc_count, term.term_freq,
                            self.index.document_count_with_term(word))
                        tf_idf_pow = math.pow(tf_idf, 2)
                        norm_list.append(tf_idf_pow)

            norm_list_sum = sum(norm_list)
            norm = math.sqrt(norm_list_sum)
            self.document_norm[document_id] = norm


class RankingModel():

    @abstractmethod
    def get_ordered_docs(
        self, query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]]
    ) -> (List[int], Mapping[int, float]):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método"
        )

    def rank_document_ids(self, documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key=lambda x: -documents_weight[x])
        return doc_ids


class OPERATOR(Enum):
    AND = 1
    OR = 2


#Atividade 1
class BooleanRankingModel(RankingModel):

    def __init__(self, operator: OPERATOR):
        self.operator = operator

    def intersection_all(
            self,
            map_lst_occurrences: Mapping[str,
                                         List[TermOccurrence]]) -> List[int]:
        # set_ids = set()
        lst_occurrences = []

        if len(list(map_lst_occurrences.values())) == 0:
            return []

        for term, term_ocurrences in map_lst_occurrences.items():
            mapping = {term: term_ocurrences}
            lst_occurrences.append(self.union_all(mapping))

        # https://stackoverflow.com/questions/3852780/python-intersection-of-multiple-lists
        return set.intersection(
            *[set(ocurrence) for ocurrence in lst_occurrences])

    def union_all(
            self,
            map_lst_occurrences: Mapping[str,
                                         List[TermOccurrence]]) -> List[int]:
        set_ids = set()

        for _, lst_occurrences in map_lst_occurrences.items():
            for term_ocurrence in lst_occurrences:
                if term_ocurrence.doc_id not in set_ids:
                    set_ids.add(term_ocurrence.doc_id)

        return set_ids

    def get_ordered_docs(
        self, query: Mapping[str, TermOccurrence],
        map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> (List[int], Mapping[int, float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences), None
        else:
            return self.union_all(map_lst_occurrences), None


#Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self, idx_pre_comp_vals: IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term: int) -> float:
        return 1 + math.log(freq_term, 2)

    @staticmethod
    def idf(doc_count: int, num_docs_with_term: int) -> float:
        return math.log((doc_count / num_docs_with_term), 2)

    @staticmethod
    def tf_idf(doc_count: int, freq_term: int, num_docs_with_term) -> float:
        tf_value = VectorRankingModel.tf(freq_term)
        idf_value = VectorRankingModel.idf(doc_count, num_docs_with_term)
        print(
            f"TF:{tf_value} IDF:{idf_value} n_i: {num_docs_with_term} N: {doc_count}"
        )
        return tf_value * idf_value

    def get_ordered_docs(
        self, query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]]
    ) -> (List[int], Mapping[int, float]):
        documents_weight = {}

        #retona a lista de doc ids ordenados de acordo com o TF IDF
        return self.rank_document_ids(documents_weight), documents_weight
