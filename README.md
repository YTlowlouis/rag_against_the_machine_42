# rag_against_the_machine_42


libraries to use:

	python fire for cli

	tqdm for loading bar


model: Qwen/Qwen3-0.6B

Indexing: 
	
Two distinct method for:
	
	python code

	markdown file

Retrieval classic lexical method:
	
	TF-IDF

	BM25

Retrieval:

	Max and default chunk size for input file: 2000
	
	returns the topk most relevant snippet ; A file path and a character range: firts letter index, last letter index

