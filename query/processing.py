from typing import List, Set, Mapping
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import BooleanRankingModel, RankingModel, VectorRankingModel, IndexPreComputedVals, OPERATOR
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner


class QueryRunner:
	def __init__(self, ranking_model: RankingModel, index: Index, cleaner: Cleaner):
		self.ranking_model = ranking_model
		self.index = index
		self.cleaner = cleaner

	def get_relevance_per_query(self) -> Mapping[str, Set[int]]:
		"""
		Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
		fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

		"""
		base_path = "relevant_docs"
		dic_relevance_docs = {}
		for location in ["belo_horizonte", "irlanda", "sao_paulo"]:
			with open(f"{base_path}/{location}.dat", encoding='utf-8') as file:
				dic_relevance_docs[location] = set(file.readline(). rstrip('\n').split(","))
		return dic_relevance_docs

	def count_topn_relevant(self, n: int, respostas: List[int], doc_relevantes: Set[int]) -> int:
		"""
		Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
		Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
		Os documentos relevantes estão no parametro docRelevantes
		"""
		# print(f"Respostas: {respostas} doc_relevantes: {doc_relevantes}")
		relevance_count = 0
		size = len(respostas)
		if n > size:
			n = size
		if respostas is not None:
			for i in range(n):
				if respostas[i] in doc_relevantes:
					relevance_count += 1
		return relevance_count

	def compute_precision_recall(self, n: int, lst_docs: List[int], relevant_docs: Set[int]) -> (float, float):

		precision = None
		recall = None
		return precision, recall

	def get_query_term_occurence(self, query:str) -> Mapping[str,TermOccurrence]:
		"""
			Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
			E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
			e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
			Coloque o docId como None.
			Caso o termo nao exista no indic, ele será desconsiderado.
		"""
		map_term_occur = {}
		for word in query.split(" "):
			pre_processed = self.cleaner.preprocess_word(word)
			if pre_processed in self.index.dic_index:
				if pre_processed not in map_term_occur:
					map_term_occur[pre_processed] = TermOccurrence(None, self.index.get_term_id(pre_processed), 1)
				else:
					map_term_occur[pre_processed].term_freq += 1
		# print(self.index)

		return map_term_occur

	def get_occurrence_list_per_term(self, terms:List) -> Mapping[str, List[TermOccurrence]]:
		"""
			Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
			Caso o termo nao exista, este termo possuirá uma lista vazia
		"""
		dic_terms = {}
		for term in terms:
			dic_terms[term] = self.index.get_occurrence_list(term)


		return dic_terms
	def get_docs_term(self, query:str) -> List[int]:
		"""
			A partir do indice, retorna a lista de ids de documentos desta consulta
			usando o modelo especificado pelo atributo ranking_model
		"""
		# Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
		dic_query_occur = self.get_query_term_occurence(query)
		print('teste', list(dic_query_occur))
		# obtenha a lista de ocorrencia dos termos da consulta
		dic_occur_per_term_query = {}
		term = []
		if(len(dic_query_occur) > 0):
			term = list(dic_query_occur)
		dic_occur_per_term_query = self.get_occurrence_list_per_term(term)	

		# utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
		sorted_documents = self.ranking_model.get_ordered_docs(dic_query_occur,dic_occur_per_term_query)
		return sorted_documents

	@staticmethod
	def runQuery(query:str, indice:Index, ranking_model: RankingModel, indice_pre_computado:IndexPreComputedVals , map_relevantes:Mapping[str,Set[int]]):
		"""
			Para um daterminada consulta `query` é extraído do indice `index` os documentos mais relevantes, considerando 
			um modelo informado pelo usuário. O `indice_pre_computado` possui valores précalculados que auxiliarão na tarefa. 
			Além disso, para algumas consultas, é impresso a precisão e revocação nos top 5, 10, 20 e 50. Essas consultas estão
			Especificadas em `map_relevantes` em que a chave é a consulta e o valor é o conjunto de ids de documentos relevantes
			para esta consulta.
		"""
		time_checker = CheckTime()

		# PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
		# apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
		# abaixo, existem exemplos fixos.
		qr = QueryRunner(indice, ranking_model(indice_pre_computado))
		time_checker.print_delta("Query Creation")


		# Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
		docs = qr.get_docs_term(query)
		respostas = None
		time_checker.print_delta("anwered with {len(respostas)} docs")

		# nesse if, vc irá verificar se a query existe no hashmap de documentos relevantes (leve processamento)
		# se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
		# O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao
		# obtem o hash map
		# faz as conversoes 

		if(True):
			arr_top = [5,10,20,50]
			revocacao = 0
			precisao = 0
			for n in arr_top:
				revocacao = 0#substitua aqui pelo calculo da revocacao topN
				precisao = 0#substitua aqui pelo calculo da revocacao topN
				print("Precisao @{n}: {precisao}")
				print("Recall @{n}: {revocacao}")

		# imprima aas top 10 respostas

	def open_index_by_path(path) -> Index:
		# TODO: Abrir index e retorna-lo
		pass

	@staticmethod
	def main(ranking_identifier, ranking_operator_identifier, index_path):
		
		# TODO: Alterar para input ser um dict contendo esses parametros como chave
		ranking_model = None
		operator = None
		index = None
		
		if(ranking_operator_identifier):
			operator = ranking_operator_identifier
		else:
			raise Exception("Necessario parametro ranking_operator_identifier")
		if(ranking_identifier):
			if(ranking_identifier is "vector"):
				ranking_model = VectorRankingModel(operator)
			elif(ranking_identifier is "boolean"):
				ranking_model = BooleanRankingModel(operator)
		else:
			raise Exception("Necessario parametro ranking_identifier")
			
	
		# leia o indice (base da dados fornecida)
		if(index_path):			
			index = QueryRunner.open_index_by_path(index_path)
		else:
			raise Exception("Necessario parametro index_path")


		# Checagem se existe um documento (apenas para teste, deveria existir)
		print(f"Existe o docId? 105047: {index.hasDocId(105047)}")

		# Instancie o IndexPreComputedVals para pre-computar os valores necessarios para a query
		print("Precomputando valores atraves do indice...")
		check_time = CheckTime()
		
		index_pre_computed = IndexPreComputedVals(index)

		check_time.print_delta("Precomputou valores")

		# encontra os docs relevantes
		map_relevance = None
		
		print("Fazendo query...")
		# aquui, peça para o usuário uma query (voce pode deixar isso num while ou fazer um interface grafica se estiver bastante animado ;)
		query = "São Paulo"
		QueryRunner.runQuery(query,index, ranking_model, index_pre_computed,map_relevance)
