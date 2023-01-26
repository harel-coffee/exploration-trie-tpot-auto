import pandas as pd
import os, sys
from scipy import stats


def extract_holdout_set_accuracy(dataset, k):

	if dataset == "anges":
		file =  "summary_data/" + dataset +"_ci.csv"
		data = pd.read_csv(file)
		nsga = data[data['type']=='NSGA-II']
		lex = data[data['type']=='Lexicase']
	else:
		file =  "summary_data/" + dataset + str(k) +"_ci.csv"
		data = pd.read_csv(file)
		nsga = data[data['type']=='baseline']
		lex = data[data['type']=='epsbestlex']

	return nsga.iloc[:, 5], lex.iloc[:, 5] # Remove the first row with all zeros


nsga, lex = extract_holdout_set_accuracy('anges', 0)
u, p = stats.mannwhitneyu(nsga, lex)
print("ANGES	", "NSGA:	", sum(nsga)/len(nsga), "LEX	", sum(lex)/len(lex), "	", p)

for k in [2, 4, 7, 14, 23, 24, 25, 27, 28, 30, 32, 35, 40]:
	nsga, lex = extract_holdout_set_accuracy('digen', k)
	u, p = stats.mannwhitneyu(nsga, lex)
	print("DIGEN-", k, "	", "NSGA:	", sum(nsga)/len(nsga), "LEX	", sum(lex)/len(lex), "	", p)



