# exploration-trie-tpot
Comparison of evolutionary algorithms within TPOT using a trie to determine their exploratory nature
![alt text](https://github.com/EpistasisLab/exploration-trie-tpot/blob/main/sample_trie.jpg?raw=true)
### Supplemental Data

After running TPOT with the included code (differs from the original TPOT as it has much more benchmarking code), we can read in the pickled TPOT run with all the evaluated pipelines within the run as a union of explored trees as a trie.

The PipelineTrie class is a data structure that represents pipelines as a trie. Each node of the trie is an instance of the TrieNode class and represents a primitive of a pipeline. The pipelines are converted from their string representation to a tree structure and then integrated into the trie. A node's data, such as the number of times it has been traversed, the pipeline's cross-validation score, and generation, are stored as class variables.

 The TrieNode is a node in the Trie data structure that holds information about a pipeline primitive (e.g., a step in the pipeline). Each node holds the following information:
- "primitive": a string representing the primitive.
- "path": a string of primitives that make up the pipeline.
- "traverse_count": the number of times this node has been visited.
- "total_cv_score": a list of cross-validation scores.
- "generation": a list of the generations the pipeline was created.
- "children": a dictionary of child nodes.
- "parents": a list of parent nodes.
- "depth": the depth of the node in the Trie.
- "max_score": the maximum cross-validation score for this node.
- "min_score": the minimum cross-validation score for this node.
- "diversity_score": a score to measure diversity of primitives.

The "insert" method adds a pipeline to the Trie. It converts the pipeline into a list, removes None values, and adds the pipeline to the Trie using a depth-first search (dfs) through the tree. For each node in the Trie, the method updates its traverse_count, total_cv_score, generation, depth, min_score, and max_score. It also updates the diversity_score of parent nodes.

## Requirements
The code was written in Python 3.9 and uses TPOT and the DEAP library.
The following libraries should be installed:
- TPOT
- DEAP
- mlinsights
- networkx

## Step 1 - Run TPOT (included)
There are two custom TPOT codebases included, one with the default NSGA-II selection method and another using the Automated Epsilon Lexicase selection method. As a guideline, follow the digen_benchmarking.py script in the respective folder to run TPOT and save the requisite benchmarking info. 
Included benchmarking info:
- fitness tracking per individual (csv format): operator count, fitness score, generation 
- offspring crossover performance per every occurrence of crossover (csv format): parent 1 fitness, parent 2 fitness, offspring fitness, generation
- offspring mutation performance per every occurrence of mutation (csv format): parent fitness, offspring fitness, generation
- pareto front models per generation (csv format): sklearn pipeline, cv score, generation, holdout score, holdout roc auc score
- resource logging (csv format): local memory usage and time usage per generation
- evaluated_individuals (pickle file): contains all the pipelines evaluated by TPOT. This file is key to build the exploration trie.

##  Step 2 - Build the exploration trie
Follow and run the digen_trie_generation.py in the main folder.
