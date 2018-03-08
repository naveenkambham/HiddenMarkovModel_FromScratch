"""
Title:Implement HMM run various work flows to understand the system.
Developer : Naveen Kambham
Description: This is a simple two state HMM model implemented using matrices. It has various workflows and methods to
             Caliculate Transition, observation matrices.

"""

"""
Importing the required libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
 PlotProabbilityDistributions(yvals,zvals,Title,Xlabel,Xlabel2): Method to plot out put distributions for each casino and each state
 [in]: Yvalues - Loose count, zvals- win count, Labels for Cheat and fair states
 [out]: Plot
"""
def PlotProabbilityDistributions(yvals,zvals,Title,Xlabel,Xlabel2):
    N = 2
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #Use Y values and colour red for lose
    rects1 = ax.bar(ind, yvals, width, color='r')
    #Use Z values and colour gree for win
    rects2 = ax.bar(ind+width, zvals, width, color='g')

    #Setting the lables for X and Y axis
    ax.set_ylabel('Probability')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( (Xlabel, Xlabel2) )
    ax.legend( (rects1[0], rects2[0]), ('Lose', 'Win') )

    #Adding Title
    fig.suptitle(Title)
    plt.show()

"""
 PlotProabbilityDistributions(yvals,zvals): Method to plot out put distributions for no of times a casino is in each state
  [in]: Yvalues - Cheat count, zvals- Fair count
 [out]: Plot
"""
def PlotOutputDistributions(yvals,zvals,):
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #Use Y values and colour red for cheat state
    rects1 = ax.bar(ind, yvals, width, color='r')
    #Use Z values and colour green for fair state
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    #Add labels
    ax.set_ylabel('Count')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('Lion', 'Dragon', 'Pyramid') )
    ax.legend( (rects1[0], rects2[0]), ('Cheat', 'Fair') )
    plt.show()

"""
Method to read the training data and add state columns in memory
[in]: Input file
[out]: data
"""
def readTrainingData(filepath):
    #Read CSV
    data = pd.read_csv(filepath, sep="\t", header=None)
    #Adding Columns
    data.columns = ["state", "outcome"]
    return data

"""
Method to read the testing data and replace win with one and lose with zero to perform the caliculations easily
[in]: Input file
[out]: data
"""
def readTestData(filepath):
    f = open(filepath,'r')
    filedata = f.read()
    f.close()
    #Replace win and loses
    newdata = filedata.replace("win","1")
    newdata2= newdata.replace("lose",'0')
    f = open(filepath,'w')
    f.write(newdata2)
    f.close()

    #Loading the data
    data= np.loadtxt(filepath)
    # print(data)
    return data

"""
ComputeTransitionMatrix: Method to compute the Transition and observation matrices.
"""
def computeTransitionMatrix(data):
     stateData = data['state']
     #Dictionary to hold the states count
     b ={}

     #Counter to count all the state pairs. Here Zip funcion is used to create a tuple of all possible pairs
     trans_counter=Counter(zip(stateData, stateData[1:]))

     #Iterating counter to add the values to dictionaries
     for (x,y), c in trans_counter.most_common():
          b[x,y] = c
     #creating transition matrix
     temptransMatrix= np.array([[(b['cheat','cheat'])/(b['cheat','fair']+b['cheat','cheat']),(b['cheat','fair'])/(b['cheat','fair']+b['cheat','cheat'])],
                                 [(b['fair','cheat'])/(b['fair','fair']+b['fair','cheat']),(b['fair','fair'])/(b['fair','fair']+b['fair','cheat'])]])

     #using a data frame to add columns and indexes
     transitionMatrixDf = pd.DataFrame(temptransMatrix,index=['cheat','fair'])
     transitionMatrixDf.columns=['cheat','fair']

     print("Transition Matrix:")
     print(transitionMatrixDf)

     #Counting States and Win Losses
     #Here also Zip funcion is used to create a tuple of all possible out comes
     obs_counter=Counter(zip(data['state'],data['outcome']))
     # print(obs_counter)

     #Dictionary to hold the observation counts
     obs ={}
     for (x,y), c in obs_counter.most_common():
          obs[x,y] = c

     # Creating Observation matrix
     obs_matrix= np.array([[(obs['cheat','lose'])/(obs['cheat','win']+obs['cheat','lose']),(obs['cheat','win'])/(obs['cheat','win']+obs['cheat','lose'])],
                      [(obs['fair','lose'])/(obs['fair','win']+obs['fair','lose']),(obs['fair','win'])/(obs['fair','win']+obs['fair','lose'])]
                         ])
     obs_matrixdf = pd.DataFrame(obs_matrix,index=['cheat','fair'])
     obs_matrixdf.columns=['lose','win']
     print("Emission Matrix:")
     print(obs_matrixdf)
     return temptransMatrix,obs_matrix


"""
This method is to compute alpha and beta values using transition and observation matrices and then preditcing the states at each possible observation.
[in]:Transtion, Observation matrices, Observations
"""
def forward_backward_alg(A_mat, O_mat, observ):

    k = observ.size
    (n,m) = O_mat.shape
    #initializing forward and backward place holders to store compute probabilities
    prob_mat = np.zeros( (n,k) )
    fw = np.zeros( (n,k+1) )
    bw = np.zeros( (n,k+1) )
    print(observ)
    # Forward step
    fw[:, 0] = 1.0/n

    #Iterating all observations
    for obs_ind in range(k):

        #Taking current row
        pi_row_vec = np.matrix(fw[:,obs_ind])

        #updating the next row given the current values
        fw[:, obs_ind+1] = pi_row_vec *(np.diag(O_mat[:,observ[obs_ind]]))* np.matrix(A_mat).transpose()
        #Normalizing the prob values
        fw[:,obs_ind+1] = fw[:,obs_ind+1]/np.sum(fw[:,obs_ind+1])

    # backward step
    bw[:,-1] = 1.0

    #Iterating all observations from back
    for obs_ind in range(k, 0, -1):
        b_col_vec = np.matrix(bw[:,obs_ind]).transpose()
        #Updating row based on next observation rows
        bw[:, obs_ind-1] = (np.matrix(A_mat) * np.matrix(np.diag(O_mat[:,observ[obs_ind-1]])) * b_col_vec).transpose()
        #Normalizing proababilities
        bw[:,obs_ind-1] = bw[:,obs_ind-1]/np.sum(bw[:,obs_ind-1])


    # combine Step
    prob_mat = np.array(fw)*np.array(bw)
    prob_mat = prob_mat/np.sum(prob_mat, 0)

    #Counter to caliculate the number of times system in each state
    cnt= Counter(prob_mat.argmax(axis=0))

    #Converting from zero and ones to Cheat and Fair
    for key,val in cnt.most_common(len(cnt)):
        if (key == 0):
            cheat= val
        else:
            fair= val

    return prob_mat, fw, bw

def main():

    #input training files
    inputfiles=[r'/home/naveen/Downloads/DataSets/training_Lion_1000.data.txt',
                ]
    #Observation Files
    testingfiles=[r'/home/naveen/Downloads/DataSets/testing_Dragon_1000.data.txt',
                 ]

    for i in range(0,1):
        print("Transition and Emission matrices for:",inputfiles[i])
        data = readTrainingData(inputfiles[i])
        A,B = computeTransitionMatrix(data)
        print(A)
        print(B)

        forward_backward_alg(A,B,readTestData(testingfiles[i]))

main()



