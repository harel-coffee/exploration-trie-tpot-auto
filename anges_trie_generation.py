import dill as pickle
import sys
import pandas as pd
from tpot import TPOTClassifier
from digen import Benchmark
from os.path import exists
from exploration_trie import PipelineTrie, extract_labels
import exploration_trie as et


directoryevs = ["/baseline/baseline","/lexicase/lexicase"]
upper_quantile_only = False

generation_num = 50
total_runs = 40

name_values = {
    "/baseline/baseline" : "baseline",
    "/lexicase/lexicase" : "lexicase",
}

result = {}

import pandas as pd
#ANGES DATASET not available for public
dataset = pd.read_csv("/Users/matsumoton/Documents/anges_cad_1_train.csv",sep=",")
y_train = dataset['target']
X_train = dataset.drop(['target'],axis=1)

test_dataset = pd.read_csv("/Users/matsumoton/Documents/anges_cad_1_test.csv",sep=",")
y_test = test_dataset['target']
X_test = test_dataset.drop(['target'],axis=1)



tpot = TPOTClassifier(verbosity=2, population_size=1, generations=1)
tpot.fit(X_train, y_train)

diversity_scores= {}

for directoryev in directoryevs:
    temp_ev = []
    #pipeline_trie = PipelineTrie()
    diversity_scores[directoryev] = []
    print(directoryev)
    for i in range(0,total_runs):
        pipeline_trie = PipelineTrie()
        pklfile = f"/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/anges{directoryev}_{i}_evaluated_individuals.pkl"
        if not exists(pklfile):
                continue
        with open(pklfile, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            result = unpickler.load()
            for k , v in result.items():
                #print(k)
                pipeline_trie.insert(k,v,tpot._pset)
        
        pipeline_trie.get_networkx_graph(100)

        pipeline_trie.display(f"/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/anges{directoryev}_run{i}_ds_{pipeline_trie.root.diversity_score}")
        print("global efficiency run " + str(i) + " : " + str(na.global_efficiency(pipeline_trie.graph)))
