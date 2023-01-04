# -*- coding: utf-8 -*-
"""
Spyder Editor

This file includes all analyses performed as part of the "Meetomes" project. After identifying six interactional roles and counting how often each of the team members produced role-specific actions, we wanted to correlate the occurrence of such actions to network measures as well as meeting evaluations. For that purpose, we use this script.
Input: table of all data, including meeting (date), name (team member), count of actions corresponding to each role, nodal network measures, and meeting evaluation survey data.
Output: table of correlation coefficients and uncorrected p-values for all nodal correlation tests; table of correlation coefficients and Bonferroni corrected p-values for global correlation tests.

"""
#%%

#0. Import packages, suppress warnings, set working directory
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import scipy.stats
from scipy.stats import pearsonr
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings('ignore')

os.chdir('C:/Users/marlo/Documents/Meetomes')

#%%

#1. Load data
data = pd.read_excel('finaldata_def.xls', header=0) #take the first row as column headers
data["local_eff"] = data["local_eff"].astype(float) #make sure all values are of type float

#%%

#2. Transform data
# Calculate total number of role-specific utterances
data['total_actions'] = data['chair']+data['skeptic']+data['expert']+data['clarifier']+data['connector']+data['practical']

# Change counts of role-specific actions to proportions relative to the total
roles = ["chair", "skeptic", "expert", "clarifier", "connector", "practical"]
for col in roles:
    data[col] = data[col] / data["total_actions"] #calculate relative proportions of role-specific actions
    data[col] = data[col].fillna(0) #omit NA; these were produced in instances where the formula above was trying to divide by zero
    
#Rename integers to letters from the alphabet (otherwise analysis will treat the variable name as an integer)
alphabet=['AA','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','ZZ']
numbers=set(range(1,28,1))
counter=0

for j in data['name']: #for-loop to replace integers by letters from the alphabet
    for i in numbers:
        if data.iloc[counter,1] == i: #refers to the name column, i.e. team members in the meeting
            data.iloc[counter,1] = alphabet[i]
    counter+=1
    
#Save outcomes in following dataframe
cols=['chair_r', 'chair_p', 'skeptic_r', 'skeptic_p', 'expert_r','expert_p','clarifier_r','clarifier_p', 'connector_r','connector_p','practical_r','practical_p']
rows=['local_eff','nod_str','partcoef','wmdegree','betwcent','effect_norm','worth_norm','overall_norm','motivation_norm','education_norm','translation_norm','connection_norm','understand_norm','mood_norm','total_eval']
output= pd.DataFrame(columns=cols,index=rows)

#%%

#3. Correlate proportions of role-specific actions to meeting evaluations

#Subset only the selection of variables that we need for correlations with meeting evaluations
q_items = ['effect_norm','worth_norm','overall_norm','motivation_norm','education_norm','translation_norm','connection_norm','understand_norm','mood_norm']
columns_for_analysis = ["name"] + roles + q_items
data_evals = data[columns_for_analysis].dropna() #dropping NAs here prevents loss of data (some ppl have meeting evaluations but no network data, and vice versa)

#Reset indices for clarity
data_evals.reset_index(inplace=True, drop=True)

#Create a dummy for each team member
dummies = pd.get_dummies(data_evals['name'])

#Merge the dummy df with the data df
data_evals = pd.concat([data_evals,dummies], axis=1)

#Rename dummy columns for clarity
for i in data_evals.iloc[:,16:]:
    data_evals.rename(columns = {str(i):'dummy_'+str(i)}, inplace = True)
    
#Create the dummy string (stringZ), which we will use as a set of predictors for subsequent regression models (which are used to calculate r from B)
Z = data_evals.iloc[:,13:].columns

stringZ = ' '
dummy_counter = 0
for dummy in Z:
    if dummy_counter == 0:
        stringZ = stringZ + str(dummy)
    else:
        stringZ = stringZ + '+ ' + str(dummy)
        
    dummy_counter+=1

#Loop over action proportions and correlate them to evaluations
for row_item in q_items: #cells containing evaluations
    for col_item in roles: #cells containing proportions
        string = "{} ~ {}".format(row_item, col_item)
        print(string)
        md = smf.mixedlm(string, data_evals, groups=data_evals["name"]) #subsets different variables in the linear mixed model
        mdf = md.fit()
        output.loc[row_item,col_item + "_p"] = mdf.pvalues[1] #save p-value, not yet corrected for multiple testing!

        #Calculate correlation coefficient r from regression coefficient B (mdf.params[1])
        #1: How much variation in X (col_item, aka roles) is explained by Z (team member, which here is the group variable)?
        residX = smf.ols(f"{col_item} ~"+stringZ, data_evals).fit().resid #input the dummy string containing all team members

        #2: How much variation in Y (row_item, aka ntwrk msr) is explained by Z (team member, which here is the group variable)?
        residY = smf.ols(f"{row_item} ~"+stringZ, data_evals).fit().resid #input the dummy string containing all team members

        #3: Calculate correlation coefficient r from regression coefficient B
        r = mdf.params[1] * residX.std() / residY.std() #B = mdf.params[1]
        output.loc[row_item,col_item + "_r"] = r # save correlation coefficient

        #Instantaneously report significant correlations
        if mdf.pvalues[1] <= 0.05:
            print('Significant correlation between '+col_item+' and '+row_item+':')
            print('Coefficient: '+ str(mdf.params[1]))
            print('R: '+ str(r))
            print('P-value: '+ str(mdf.pvalues[1]))
            print('Bonferroni-corrected P: '+ str(mdf.pvalues[1].astype(float) * 48)) #6 roles * 8 questionnaire items = 48 tests
    
#4. Correlate proportions of role-specific actions to network measures
#Subset only the data that you need for these analysis and drop NA values
data_ntwrk = data.iloc[:,0:13].dropna() #dropping NAs here prevents loss of data (some ppl have meeting evaluations but no network data, and vice versa)

#Reset indices for clarity
data_ntwrk.reset_index(inplace=True, drop=True)

#Create Create a dummy for each team member
dummies = pd.get_dummies(data_ntwrk['name'])

#Merge the dummy df with the data df
data_ntwrk = pd.concat([data_ntwrk,dummies], axis=1)

#Rename dummy columns for clarity
for i in data_ntwrk.iloc[:,13:]:
    data_ntwrk.rename(columns = {str(i):'dummy_'+str(i)}, inplace = True)

#Generate string to use in regression formula later (required for converting B coefficient to correlation coefficient r)
Z = data_ntwrk.iloc[:,13:].columns

stringZ = ' '
dummy_counter = 0
for dummy in Z:
    if dummy_counter == 0:
        stringZ = stringZ + str(dummy)
    else:
        stringZ = stringZ + '+ ' + str(dummy)
        
    dummy_counter+=1

#Create variable names that should be iteratively used as input (same as column names corresponding to desired variables)
ntwrkmsrs = ['local_eff','nod_str','partcoef','wmdegree','betwcent']

# Loop over action proportions and correlate them to network measures
for row_item in ntwrkmsrs: #cells containing network measures
    for col_item in roles: #cells containing action proportions
        string = "{} ~ {}".format(row_item, col_item)
        print(string)
        md = smf.mixedlm(string, data_ntwrk, groups=data_ntwrk["name"]) #create linear mixed model, iteratively subset variables
        mdf = md.fit() #model fit parameters
        output.loc[row_item,col_item + "_p"] = mdf.pvalues[1] #save uncorrected p-value

        #Calculate correlation coefficient r from regression coefficient B
        #1: How much variation in X (col_item, aka roles) is explained by Z (team member, which here is group variable)?
        residX = smf.ols(f"{col_item} ~"+stringZ, data_ntwrk).fit().resid
        
        #2: How much variation in Y (row_item, aka ntwrk msr) is explained by Z (team member, which here is group variable)?
        residY = smf.ols(f"{row_item} ~"+stringZ, data_ntwrk).fit().resid

        #3: Calculate correlation coefficient r from regression coefficient B
        r = mdf.params[1] * residX.std() / residY.std() #B = mdf.params[1]
        output.loc[row_item,col_item + "_r"] = r # save correlation coefficient
        
        #Instantaneously report significant correlations
        if mdf.pvalues[1] <= 0.05:
            print('Significant correlation between '+col_item+' and '+row_item+':')
            print('Coefficient: '+ str(mdf.params[1]))
            print('R: '+ str(r))
            print('P-value: '+ str(mdf.pvalues[1]))
            print('Bonferroni-corrected P: '+ str(mdf.pvalues[1].astype(float) * 30)) #6 roles * 5 network measures = 30 tests

#Save output file
output.to_excel("output.xlsx")
            
#%%

#6. Create a heatmap of the nodal network measure correlations
#Put the correlation coefficients that you want to plot in a matrix
r_values=['chair_r','skeptic_r','expert_r','clarifier_r','connector_r','practical_r']
mat = output.loc['local_eff':'betwcent',r_values].astype(float) #subset of the output

#Generate plot
netwmeasures = ['local efficiency','nodal strength','participation coefficient','within-module degree','betweenness centrality']
actiontypes = ['chair','skeptical','expert','clarifying','connecting', 'practical']
plot = sns.heatmap(mat, xticklabels=actiontypes, yticklabels=netwmeasures, center = 0, cmap='magma')
plot.set(xlabel="Interactional role", ylabel="Network measure")
plot.set_title('Correlations between interactional roles and network measures')

#make sure that ticks are shown diagonally above the figure
plt.setp(plot.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.show()

#%%

# Calculate correlations with global measures
globaldata = pd.read_excel('globaldata.xlsx')

#Make correlation matrix
globmat = globaldata.corr()

#Calculate p-values
def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues

pvals = calculate_pvalues(globmat) * 12 #Bonferroni correction, 2 global netw measures * 6 roles

#Reshape output
globmat = globmat.iloc[6:8,0:6] #remove autocorrelations and duplicate correlation pairs
pvals = pvals.iloc[6:8,0:6] #do the same for the corresponding pvalues

globmat.loc['modularity_r'] = globmat.loc['modularity'] #copy modularity coefs to a third row
globmat.rename(index={'modularity':'GE_norm_p'}, inplace=True) #rename row
globmat.loc['modularity_p'] = pvals.loc['modularity'] #add modularity p-values to global matrix
globmat.loc['GE_norm_p'] = pvals.loc['GE_norm'] #transfer p-values to row underneath GE coefs

#Save output files
globmat.to_excel("global_corrcoefs.xlsx")
