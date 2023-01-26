import pandas as pd
import os, sys
from scipy import stats


def extract_convergence_points(out_dir):
	#empty df for results

	results = pd.DataFrame([[0,0,0,0,0,0]], columns = ['run', 'max', 'thresh', 'ops', 'score','gen'])

	#range should be however many files in that folder, but had tested few..

	for file in os.listdir(out_dir):
		if file.endswith("_fitness.csv") and not file.endswith("pareto_fitness.csv"):
			run = int(file.split("_")[-2])

			file = out_dir + "/" + file
			data = pd.read_csv(file)
			
			
			final_gen_data = data[data['generation']==99]

			maximum_val = max(final_gen_data['score'])

			thresh = .99*maximum_val 

			thresholded = data[data['score'] >= thresh]

			minimum_val = min(thresholded['generation'])

			df = thresholded.query('generation == @minimum_val')

			df2 = pd.DataFrame([[run, maximum_val,thresh, df['operator_count'].iloc[0],df['score'].iloc[0],df['generation'].iloc[0]]],columns = ['run', 'max', 'thresh', 'ops', 'score','gen'])

			results = pd.concat([results, df2])

	#results.iloc[1:, ].to_csv("summary_data/"+out_dir[6:]+".csv")

	return results.iloc[1:, 3] # Remove the first row with all zeros

#lex = extract_convergence_points("anges/lexicase_gen100")
nsga = extract_convergence_points("anges/baseline_gen100")
ebest = extract_convergence_points("anges/epsbestlexicase_gen100")
#u1, p1 = stats.mannwhitneyu(lex, nsga)
#u2, p2 = stats.mannwhitneyu(lex, ebest)
u3, p3 = stats.mannwhitneyu(nsga, ebest)
print(p3)
print("ANGES	", sum(nsga)/len(nsga), "	", sum(ebest)/len(ebest))

