::::::::::::::::::::::: University of Trento - Italy ::::::::::::::::::::::::::::::::

:::::::::::: SICK (Sentences Involving Compositional Knowledge) data set ::::::::::::

:::::::::::::::::::::http://clic.cimec.unitn.it/composes/sick/ ::::::::::::::::::::::


The SICK data set consists of 10,000 English sentence pairs, built starting from two existing 
paraphrase sets: the 8K ImageFlickr data set (http://nlp.cs.illinois.edu/HockenmaierGroup/data.html) 
and the SEMEVAL-2012 Semantic Textual Similarity Video Descriptions data set 
(http://www.cs.york.ac.uk/semeval-2012/task6/index.php?id=data). Each sentence pair is annotated 
for relatedness in meaning and for the entailment relation between the two elements.



The SICK data set is released under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 
Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US)

When using SICK in published research, please cite:
M. Marelli, S. Menini, M. Baroni, L. Bentivogli, R. Bernardi and R. Zamparelli. 2014. A SICK cure 
for the evaluation of compositional distributional semantic models. Proceedings of LREC 2014, 
Reykjavik (Iceland): ELRA. 



The SICK data set is used in SemEval 2014 - Task 1: Evaluation of compositional distributional 
semantic models on full sentences through semantic relatedness and textual entailment



File Structure: tab-separated text file

Fields:

- pair_ID: sentence pair ID

- sentence_A: sentence A

- sentence_B: sentence B

- entailment_label: textual entailment gold label (NEUTRAL, ENTAILMENT, or CONTRADICTION)

- relatedness_score: semantic relatedness gold score (on a 1-5 continuous scale)

- entailment_AB: entailment for the A-B order (A_neutral_B, A_entails_B, or A_contradicts_B)

- entailment_BA: entailment for the B-A order (B_neutral_A, B_entails_A, or B_contradicts_A)

- sentence_A_original: original sentence from which sentence A is derived

- sentence_B_original: original sentence from which sentence B is derived

- sentence_A_dataset: dataset from which the original sentence A was extracted (FLICKR vs. SEMEVAL)

- sentence_B_dataset: dataset from which the original sentence B was extracted (FLICKR vs. SEMEVAL)

- SemEval_set: set including the sentence pair in SemEval 2014 Task 1 (TRIAL, TRAIN, or TEST)
