# exploration-trie-tpot
Comparison of evolutionary algorithms within TPOT using a trie to determine their exploratory nature

This code is for a Trie-based data structure for TPOT pipelines. After running TPOT with the included code (differs from the original TPOT as it has much more benchmarking). The TrieNode is a node in the Trie data structure that holds information about a pipeline primitive (e.g., a step in the pipeline). Each node holds the following information:
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
