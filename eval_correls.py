# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

#0. Import packages, suppress warnings, set working directory
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import scipy.stats
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings('ignore')

os.chdir('C:/Users/marlo/Documents/Meetomes')

#%%

#1. Load data (header=0 takes the first row as column names)
data = pd.read_excel('finaldata.xlsx', header=0)

#Take out rows that contain missing data
data_new = data#.dropna()

#%%
#2. Transform data
# Calculate total number of role-specific utterances
data_new['total'] = data_new['chair']+data_new['skeptic']+data_new['expert']+data_new['clarifier']+data_new['connector']+data_new['practical']

# Change counts of role-specific actions to proportions relative to the total
columns_to_change = ["chair", "skeptic", "expert", "clarifier", "connector", "practical"]
for col in columns_to_change:
    data_new[col] = data_new[col] / data_new["total"]
    data_new[col] = data_new[col].fillna(0)

#%%
#3. Correlate proportions of role-specific actions to meeting evaluations

#Variables to correlate to each other
roles=columns_to_change #interactional roles
q_items = ['effect_norm','worth_norm','overall_norm','motivation_norm','education_norm','translation_norm','connection_norm','understand_norm','mood_norm']
columns_for_analysis = ["name"] + roles + q_items + ['total']
data_new = data_new[columns_for_analysis].dropna()

#%%
#Lists for outcomes
coefs = [] # make into array
pvals = []

# Loop over hub proportions and correlate them to evaluation outcomes
counter=0
counter2=0
for item in q_items: #cells containing evaluation outcomes
    for prop in roles: #cells containing proportions
        string = "{} ~ {}".format(item, prop) #Predict questionnaire item from hub proportion
        md = smf.mixedlm(string, data_new, groups=data_new["name"])
        mdf = md.fit()
        #print(string)
        #print(mdf.params)
        #print(mdf.pvalues)
        coefs.append(mdf.params[1])
        pvals.append(mdf.pvalues[1])
        if mdf.pvalues[1] <= 0.05:
            counter2+=1
            print('Significant correlation between '+prop+' and '+item+':')
            print('Coefficient: '+ str(mdf.params[1]))
            print('P-value: '+ str(mdf.pvalues[1]))
            plt.rcParams["figure.figsize"] = [16,4]
            plt.subplot(1,3,counter2+1)
            sns.regplot(data=data_new,x=prop,y=item).figure.savefig("roles_evaluations_correls.png")
        counter+=1
        
#%%

#4. Correlate proportions of role-specific actions to overall ratings of the meeting
#Sum all evaluations together -> overall rating
data_new['total_eval'] = data_new[q_items].sum(axis=1)

counter=0
for prop in roles:
    md_tot = smf.mixedlm("total_eval ~ {}".format(prop), data_new, groups=data_new["name"])
    mdf_tot = md_tot.fit()
    coefs.append(mdf_tot.params[1])
    pvals.append(mdf_tot.pvalues[1])
    if mdf.pvalues[1] <= 0.05:
        print('Significant correlation with '+prop+':')
        print('Coefficient: '+ str(mdf.params[1]))
        print('P-value: '+ str(mdf.pvalues[1]))
    counter+=1
