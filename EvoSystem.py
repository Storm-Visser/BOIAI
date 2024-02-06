import numpy as np
import pandas as pd
import math
import random

from LinReg import LinReg

def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, UseReplacementSelection, UseFitnessSelection, BitstringLength, MutationRate, CrossoverRate, PopSize, AmountOfParents):
    Pop = InitPop(BitstringLength, PopSize)
    # print(Pop)
    Results = []
    regressor = LinReg()
    data = pd.read_csv("Data/Data.txt", header=None)

    for _ in range(AmountOfGen):
        #get the top X amount of parents
        Parents, PopFitness = Selection(Pop, UseLinReg, AmountOfParents)
        # use crossover to create children
        Children = Crossover(Parents, CrossoverRate)
        # mutate some of the children
        ChildrenM = Mutate(Children, MutationRate)
        # select survivors for next gen
        NewPop = PopFitness
        if UseCrowding: #survivor selection 3 Crowding stuff
            Match(ChildrenM)
            Compete(ChildrenM)
        elif UseFitnessSelection: #survivor selection 1 Add x children to existing pop, select top fitness
            pass
        elif UseReplacementSelection: #survivor selection 2 Replace bot X of population with the new children
            pass

        Results.append(SaveResults(NewPop))
    return


def InitPop(BitstringLength, PopSize):
    # Initialize an empty list to hold the population of bitstrings
    population = []
    
    # Generate PopSize bitstrings
    for _ in range(PopSize):
        # Generate a bitstring of BitstringLength bits
        bitstring = ''.join(random.choice('01') for _ in range(BitstringLength))
        # Add the generated bitstring to the population
        population.append(bitstring)
    
    # Return the generated population
    return population

def Selection(Pop, UseLinReg, AmountOfParents):
    Selected = dict()
    for Individual in Pop:
        if UseLinReg:
            Fitness = fitnessML(Individual)
        else: 
            Fitness = fitnessSine(Individual)
        # Add the individual and its value to the dict
        Selected[Individual] = Fitness
    # Sort the dictionary to its values
    SelectedSorted = sorted(Selected.items(), key=lambda x: x[1], reverse=True)
    #Save the top X parents
    Parents = [item[0] for item in SelectedSorted[:AmountOfParents]]
    return Parents, SelectedSorted

def Crossover(Parents, Rate):
    Children = []
    checkedParents = []
    for Parent in Parents:
        # break when 10 children are found
        if len(Children) >= len(Parents): break
        # dont add parents that were already used
        if Parent in checkedParents:
            continue
        # update the used parents
        checkedParents.append(Parent)
        # Randomly select another parent that hassnt been used for crossover
        OtherParent = random.choice([x for x in Parents if x not in checkedParents])
        # update the used parent
        checkedParents.append(OtherParent)
        # Check if crossover should occur based on the rate
        if random.random() < Rate:
            # Randomly select a crossover point
            crossover_point = random.randint(1, len(Parents) - 1)

            # Perform crossover to create two children
            Child1 = Parent[:crossover_point] + OtherParent[crossover_point:]
            Child2 = OtherParent[:crossover_point] + Parent[crossover_point:]
        else:
            # If no crossover, the children are copies of the parents
            Child1 = Parent
            Child2 = OtherParent

        Children.extend([Child1, Child2])

    return Children

def Mutate(Children, Rate):
    if len(Children) <= 0: raise Exception("Children array empty")
    returnChildren = []
    for Child in Children:
        newChild = Child
        if random.random() < Rate:
            i = random.randint(0, len(Child) - 1)
            newChild = [*Child]
            newChild[i] = str((int(Child[i]) + 1) % 2)
            newChild = "".join(newChild)
        returnChildren.append(newChild)
    return returnChildren

def Match(Pop):
    return[]

def Compete(Pop):
    return[]

def SaveResults(Pop):
    return[]

def fitnessML(regressor: LinReg, data:pd.DataFrame, bitstring:str):
    X = regressor.get_columns(data.values, bitstring)
    return regressor.get_fitness(X[:,:-1], X[:,-1])

def fitnessSine(bitstring):
    bit_value = int(bitstring, 2) / (2 ** len(bitstring) - 1)
    return np.sin(bit_value *128)
