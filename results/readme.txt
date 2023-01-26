-/anges
	-/autoepslexicase_gen100 		
	-/baseline_gen100				
	-/lexicase_gen100	


	
above folders have these files:			
		{selection}_{runid}_evaluated_individuals.pkl
			-pickle file for all evaluated individuals in the tpot run. includes objective scores
		{selection}_{runid}_fitness.csv
			-metrics for each generations population
		{selection}_{runid}_metrics.csv
			-metrics generated from the trie network
		{selection}_{runid}_pareto_fitness.csv
			-pareto front models and details per generation

		disregard:	
		{selection}_{runid}_mutation_rates.csv 
		{selection}_{runid}_resources.csv_memory.csv
		{selection}_{runid}_resources.csv_time.csv


-/digen
	-/results_pop40_gen20_autolex_final
	-/results_pop40_gen20_baseline_final
	-/results_pop40_gen20_baseline_final
		

above folders have these subfolders
		-/gen_fitnesses
			-/digen{digen id}_run_{runid}_evolution_pop40_gen20.csv
				-metrics for each generations population
		-/pareto_fitnesses
			-/digen{digen id}_run_{runid}_evolution_pop40_gen20.csv
				-pareto front models and details per generation
		-/pipelines
			-/digen{digen id}_run_{runid}_metrics.csv
				-metrics generated from the trie network
			-/digen{digen id}_run_{runid}_evaluated_individuals.pkl
				-pickle file for all evaluated individuals in the tpot run. includes objective scores
		-/trie_network
			-digen{digen id}_run_{runid}_evolution_pop40_gen20.csv
			


		disregard:
		-/offspring_generation_test
		-/pareto_ci
		-/resource_logging