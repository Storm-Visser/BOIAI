import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

from LinReg import LinReg

def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, UseReplacementSelection, UseFitnessSelection, BitstringLength, MutationRate, CrossoverRate, PopSize, AmountOfParents):
    Pop = InitPop(BitstringLength, PopSize)
    # print(Pop)
    Results = []
    regressor = LinReg()
    data = pd.read_csv("Data/Data.txt", header=None)

    for _ in range(AmountOfGen):
        #get the top X amount of parents
        Parents, PopFitness = Selection(Pop, UseLinReg, AmountOfParents, regressor, data, Seed)
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
            NewPop = FitnessSelection(PopFitness, ChildrenM, UseLinReg, PopSize, regressor, data, Seed)
        elif UseReplacementSelection: #survivor selection 2 Replace bot X of population with the new children
            pass
        Pop = [x[0] for x in NewPop]
        Results.append(SaveResults(NewPop))
        print(SaveResults(NewPop))
    CreateGraph(Results) 
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

def Selection(Pop, UseLinReg, AmountOfParents, Regressor, Data, Seed):
    Selected = []
    for Individual in Pop:
        if UseLinReg:
            Fitness = fitnessML(Regressor, Data, Individual, Seed)
        else: 
            Fitness = fitnessSine(Individual)
        # Add the individual and its value as a tuple to the list
        Selected.append((Individual, Fitness))
    # Sort the list by the fitness values
    SelectedSorted = sorted(Selected, key=lambda x: x[1], reverse=False)
    # Save the top X parents
    Parents = [item[0] for item in SelectedSorted[:AmountOfParents]]
    return Parents, SelectedSorted

def Crossover(Parents, Rate):
    Children = []
    checkedParents = Parents.copy()
    for Parent in Parents:
        # break when enough children are found
        if len(Children) >= len(Parents): break
        # dont add parents that were already used
        if Parent not in checkedParents:
            continue
        # update the used parents
        checkedParents.remove(Parent)
        # Randomly select another parent that hassnt been used for crossover
        OtherParent = random.choice([x for x in Parents if x in checkedParents])
        # update the used parent
        checkedParents.remove(OtherParent)
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

def FitnessSelection(Pop, Children, UseLinReg, PopSize, Regressor, Data, Seed):
    # get the fitness of the children
    FitnessChildren = []
    for Child in Children:
        if UseLinReg:
            Fitness = fitnessML(Regressor, Data, Child, Seed)
        else: 
            Fitness = fitnessSine(Child)
        #Add the individual and its value to the dict
        FitnessChildren.append((Child, Fitness))
    # add the dicts together
    TotalFitness = Pop + FitnessChildren
    # sort them by fitness
    SortedTotalFitness = sorted(TotalFitness, key=lambda x: x[1], reverse=False)
    # get the top amount of population
    NewPop = SortedTotalFitness[:PopSize]
    return NewPop

def ReplacementSelection(Pop, Children, UseLinReg, Regressor, Data, Seed):
    childrenToAdd = []
    fitnessVal = None
    for Child in Children:
        if UseLinReg:
            fitnessVal = fitnessML(Regressor, Data, Child, Seed)
        else: 
            fitnessVal = fitnessSine(Child)
        childrenToAdd.append((Child, fitnessVal))
        
    Pop[-len(Children):] = childrenToAdd
    return Pop

def Match(Pop):
    return[]

def Compete(Pop):
    return[]


def SaveResults(Pop):
    Values = Pop
    HighestValue = Values[0][1]
    total = 0
    for Value in Values:
        total += Value[1]
    AverageValue = total / len(Values)
    return [HighestValue, AverageValue]

def fitnessML(regressor: LinReg, data:pd.DataFrame, bitstring:str, Seed):
    X = regressor.get_columns(data.values, bitstring)
    return regressor.get_fitness(X[:,:-1], X[:,-1], Seed)

def fitnessSine(bitstring):
    bit_value = int(bitstring, 2) / (2 ** len(bitstring) - 1)

    sin_value = np.sin(bit_value * 128)

    error = (2 - (sin_value + 1))/2

    return error

def CreateGraph(data):
    # Extract the values for each column (Array 1, Array 2, Array 3)
    array1_values = [sublist[0] for sublist in data]
    array2_values = [sublist[1] for sublist in data]

    # Create a list of indices (x-values) to use as the x-axis
    indices = range(len(data))

    # Create a single graph for all points
    plt.figure(figsize=(10, 6))
    plt.plot(indices, array1_values, label='Higest')
    plt.plot(indices, array2_values, label='Average')

    plt.title('All Arrays')
    plt.xlabel('Generations')
    plt.ylabel('Values')
    plt.legend()

    plt.tight_layout()
    plt.show()