import os
import sys
sys.path.insert(1,'/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmarking')
from tpot.tpot import TPOTClassifier

import pandas as pd
dataset = pd.read_csv("/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/anges/anges_cad_1_train.csv",sep=",")
y_train = dataset['target']
X_train = dataset.drop(['target'],axis=1)

test_dataset = pd.read_csv("/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/anges/anges_cad_1_test.csv",sep=",")
y_test = test_dataset['target']
X_test = test_dataset.drop(['target'],axis=1)

tpot = TPOTClassifier(verbosity=2, population_size=1, generations=1,test_x = X_test, test_y = y_test, scoring="balanced_accuracy", cv=10, dynamic_rates = True)
tpot.fit(X_train, y_train)


import dill as pickle
import pandas as pd

import numpy as np

from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle

from deap import creator
plt.rcParams["figure.figsize"] = (30,16)


directoryevs = ["/baseline/baseline","/baseline_dynamic/baseline_dynamic","/lexicase/lexicase","/lexicase_dynamic/lexicase_dynamic"]
directoryevs = ["/baseline/baseline","/lexicase/lexicase"]
upper_quantile_only = False

generation_num = 50
total_runs = 40

name_values = {
    "/baseline/baseline" : "baseline",
    "/baseline_dynamic/baseline_dynamic" : "baseline_dynamic",
    "/lexicase/lexicase" : "lexicase",
    "/lexicase_dynamic/lexicase_dynamic" : "lexicase_dynamic"
}

ev = []


for directoryev in directoryevs:
    temp_ev = [] 
    print(directoryev)
    upper_ci = []
    lower_ci = []
    test_score = []
    generations = []
    holdout_score = []
    holdout_roc_auc_score = []
    for i in range(40):
        print(i)
        #with open(f"/Users/matsumoton/Git/results_pop40_gen15_{directoryev}/pipelines/digen{j}_run_{i}_evaluated_individuals.pkl", 'rb') as file:
            #unpickler = pickle.Unpickler(file)
            #result = unpickler.load()
        ev_df_name = f"/Users/matsumoton/Library/CloudStorage/Box-Box/tpot_benchmark_data/anges{directoryev}_{i}_pareto_fitness.csv"
        #ev_df_name = f"/Users/matsumoton/Git/results_pop40_gen20_{directoryev}/pareto_fitnesses/digen{j}_run_{i}_evolution_pop40_gen20.csv"
        if not exists(ev_df_name):
            continue
        fitness_df = pd.read_csv(ev_df_name, sep=',')
        fitness_df = fitness_df.sort_values(by=['pipeline'])
        prev_pipeline = ''
        for k in range(fitness_df.shape[0]):
            pipeline = fitness_df["pipeline"][k]
            if prev_pipeline == pipeline:
                    generations.append(fitness_df["generation"][k])
                    test_score.append(p)
                    upper_ci.append(p+1.96*s)
                    lower_ci.append(p-1.96*s)
                    holdout_score.append(fitness_df["holdout_score"][k])
                    holdout_roc_auc_score.append(fitness_df["holdout_roc_auc_score"][k])
                    continue
            prev_pipeline = pipeline
            #print(pipeline)
            test = creator.Individual.from_string(pipeline, tpot._pset)
            #print(test)
            pipeline_fitted = tpot._toolbox.compile(expr=test)
            pipeline_fitted.fit(X_train, y_train)
            predictions = pipeline_fitted.predict(X_test)
            p = sum(pipeline_fitted.predict(X_test) == y_test)/len(y_test)
            s = np.sqrt(p*(1-p)/len(y_test))
            generations.append(fitness_df["generation"][k])
            test_score.append(p)
            upper_ci.append(p+1.96*s)
            lower_ci.append(p-1.96*s)
            holdout_score.append(fitness_df["holdout_score"][k])
            holdout_roc_auc_score.append(fitness_df["holdout_roc_auc_score"][k])

    con = pd.DataFrame(np.stack((generations, test_score,upper_ci,lower_ci,holdout_score,holdout_roc_auc_score), axis=1))
    con.columns = ['generations', 'test_score','upper_ci','lower_ci','holdout_score','holdout_roc_auc_score']
    con["type"] = name_values[directoryev]
    con.to_csv(f"/Users/matsumoton/pareto/{name_values[directoryev]}_anges_ci.csv", sep=',', index=False)
        

                

    
    #median normalized
#    for i in range(0,15):
#        median_gen = statistics.median(frame_df.loc[(frame_df['type']=='baseline')&(frame_df['generation']==i)]['score'])
#        frame_df.loc[frame_df['generation']==i,'score']=frame_df.loc[frame_df['generation']==i]['score'].div(median_gen)

    #for directoryev in directoryevs:
        #seaborn.violinplot(x="generation",y="score",hue="type",data=frame_df, label = "type" if i == 0 else "")
        #plt.show()
        #ax = sns.boxplot(x="generation",y="score",hue="type",data=frame_df)
        #plt.show()
        #ax = sns.swarmplot(x="generation",y="score",hue="type",data=frame_df,color=".25")

    #plt.show()

