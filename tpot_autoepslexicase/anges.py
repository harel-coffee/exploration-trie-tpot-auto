import os
os.chdir('/home/matsumoton/common/git/tpot_benchmarking_autoepslexicase')
print(os.getcwd())
from tpot.tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from test_utils import extract_labels, get_optimizer, create_dirs

import dill as pickle

import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
 # This is done based on the dataset ID.
#dataset = openml.datasets.get_dataset(1164)
#dataset = openml.datasets.get_dataset(1164)
dataset = pd.read_csv("/home/matsumoton/common/anges_cad_1_train.csv",sep=",")
y_train = dataset['target']
X_train = dataset.drop(['target'],axis=1)

test_dataset = pd.read_csv("/home/matsumoton/common/anges_cad_1_test.csv",sep=",")
y_test = test_dataset['target']
X_test = test_dataset.drop(['target'],axis=1)


#test_img = X_test
#train_img = X_train
#from sklearn.decomposition import PCA
#pca = PCA(svd_solver='randomized', iterated_power= 5)
#pca = PCA(n_components = train_img.shape[0])
#pca.fit(train_img)
#train_img = pca.transform(train_img)
#test_img = pca.transform(test_img)

#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.75, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#tpot.dump_fitness_tracker('digen25.csv')
save_directory = "/home/matsumoton/common/anges/selbest_gen100"
for run_id in range(40,50):
        # tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
        tpot = TPOTClassifier(verbosity=2, population_size=100, offspring_size=50, generations=100, track_fitnesses=True,
                track_generations=True, resource_logging=True, test_x = X_test, test_y = y_test, scoring="balanced_accuracy",cv=10, dynamic_rates = False) 
        #tpot.fit(X_train, y_train)
        tpot.fit(X_train, y_train)
        tpot.dump_fitness_tracker(f"{save_directory}_{run_id}_fitness.csv")
        tpot.dump_pareto_fitness_tracker(f"{save_directory}_{run_id}_pareto_fitness.csv")
        tpot.dump_primitives_mutations(f"{save_directory}_{run_id}_mutation_rates.csv")
        tpot.dump_resource_logging(f"{save_directory}_{run_id}_resources.csv")

        with open(f"{save_directory}_{run_id}_evaluated_individuals.pkl", 'wb') as outp:
            pickle.dump(tpot.evaluated_individuals_, outp, -1)
        
        print(save_directory)
        print(tpot.score(X_test, y_test))

