If you use this dataset, please cite this paper:


	SciTail: A Textual Entailment Dataset from Science Question Answering
	Tushar Khot and Ashish Sabharwal and Peter Clark

        @inproceedings{scitail,
                Author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
                Booktitle = {AAAI}
                Title = {SciTail: A Textual Entailment Dataset from Science Question Answering},
                Year = {2018}
                }

We release the SciTail dataset in various formats for ease of use. 

snli_format: JSONL format used by SNLI with a JSON object corresponding to each entailment example
             in each line. 
tsv_format: Tab-separated format with three columns:
            premise	hypothesis	label 
dgem_format: Tab-separated format used by the DGEM model:
             premise	hypothesis	label	hypothesis graph structure
predictor_format: JSONL format used for the AllenNLP predictors for all the entailment models with fields:
             gold_label, sentence1, sentence2, sentence2_structure 

Individual folders contain additional information about these formats. We also provide the complete
list of annotations in all_annotations.tsv with the following columns:

Question:	Original question
Answer choice:	Correct answer choice
KB Sentence:	Retrieved sentence used as the premise
Q+A as Sentence:Question and Answer choice converted into a sentence. "???" used if no ___ or wh-word found in the question.
Question Source:Source of this question from {Pub4, Pub8, SciQ}{Train, Dev, Test} 
Num. Support:	Number of crowd-workers that annotated this sentence as supporting
Num. Partial:	Number of crowd-workers that annotated this sentence as partially supporting
Num. None:	Number of crowd-workers that annotated this sentence as unrelated
Total:		Number of crowd-workers that annotated this sentence
IR Position:	Position of this sentence in the retrieved sentences for this question
Label:		Final entailment label for the premise=KB Sentence and hypothesis=Q+A as Sentence from {entails, neutral}


