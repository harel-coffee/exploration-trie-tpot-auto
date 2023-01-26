import pandas as pd
import os, sys
from scipy import stats


def extract_convergence_points(k, sel):
	#empty df for results

	out_dir = "digen/results_pop40_gen20_" + sel + "_final/gen_fitnesses"
 
	results = pd.DataFrame([[0, 0,0,0,0,0]], columns = ['run','max', 'thresh', 'ops', 'score','gen'])

	#range should be however many files in that folder, but had tested few..

	for file in os.listdir(out_dir):
		if file.startswith("digen"+str(k)+"_"):

			run = int(file.split("_")[-4])

			file = out_dir + "/" + file
			data = pd.read_csv(file)

			#print(file)
			#print(data)

			final_gen_data = data[data['generation']==19]

			maximum_val = max(final_gen_data['score'])

			thresh = .99*maximum_val 

			thresholded = data[data['score'] >= thresh]

			minimum_val = min(thresholded['generation'])

			df = thresholded.query('generation == @minimum_val')

			df2 = pd.DataFrame([[run, maximum_val,thresh, df['operator_count'].iloc[0],df['score'].iloc[0],df['generation'].iloc[0]]],columns = ['run','max', 'thresh', 'ops', 'score','gen'])

			results = pd.concat([results, df2])

	#results.iloc[1:, ].to_csv("summary_data/digen_"+str(k)+"_"+sel+".csv")

	return results.iloc[1:, 3] # Remove the first row with all zeros


for k in [2, 4, 7, 14, 23, 24, 25, 27, 28, 30, 32, 35, 40]:
	#lex = extract_convergence_points(k, "lexicase")
	nsga = extract_convergence_points(k, "baseline")
	ebest = extract_convergence_points(k, "epsbestlex")
	#print(sum(b)/len(b))
    #t, p = stats.ttest_ind(a, b)
	#u1, p1 = stats.mannwhitneyu(lex, nsga)
	#u2, p2 = stats.mannwhitneyu(lex, ebest)
	u3, p3 = stats.mannwhitneyu(nsga, ebest)
	#print("Digen-", "	", p3)
	print("Digen-",k, "	", sum(nsga)/len(nsga), "	", sum(ebest)/len(ebest))


