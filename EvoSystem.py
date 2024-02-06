import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from LinReg import LinReg

def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, UseReplacementSelection, UseFitnessSelection, BitstringLength, MutationRate, CrossoverRate, PopSize, AmountOfParents):
    Pop = InitPop(BitstringLength, PopSize)
    # print(Pop)
    Results = []
    regressor = LinReg()
    data = pd.read_csv("Data/Data.txt", header=None)
    PopEachGeneration = []

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
            NewPop = ReplacementSelection(PopFitness, ChildrenM, UseLinReg)
        Pop = [x[0] for x in NewPop]
        Results.append(SaveResults(NewPop))
        #print(SaveResults(NewPop))
        PopEachGeneration.append(NewPop)
    # CreateGraph(Results) 
    CreateSineGraph(PopEachGeneration)
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

def CreateSineGraph(generationData):

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='blue', marker='o', linestyle='', markersize=3, label='Individuals')
    sine_line, = ax.plot([], [], color='red', linewidth=0.5, alpha=0.5, label='Sine Function')
    ax.set_xlabel('Individuals')
    ax.set_ylabel('Fitness')
    ax.set_title('Population Fitness and Sine Function')
    ax.set_xlim(0, 128)
    ax.set_ylim(-1, 1)
    # ax.legend()

    # x_values = []
    # fitness_values = []
    # for generation in generationData:
        
    #     x_values = [int(x, 2) / (2 ** len(x) - 1) for x, y in generation]
    #     fitness_values = [y for x, y in generation]

    # def update(frame):
    #     line.set_data(x_values, fitness_values)
    #     return


    # Generate the initial population of individuals
    # population_size = 100
    # bitstrings = np.random.randint(0, 2, size=(population_size, 10))  # Assuming each individual has a bitstring of length 10


    # Function to update the plot with each frame (generation)
    def update(frame):
        # Compute the fitness of each individual using the sine function
        # x_values = np.arange(population_size)  # Use indices of individuals as x values
        # fitness_values = np.sin(np.sum(bitstrings, axis=1))  # Compute fitness based on the sum of bits in each bitstring

        x_values = [(int(x, 2) / (2 ** len(x) - 1)) * 128 for x, y in generationData[frame]]
        fitness_values = [y for x, y in generationData[frame]]

        print("x_values", x_values)
        print("fitness_values", fitness_values)

        # Update the data for the individuals scatter plot
        line.set_data(x_values, fitness_values)

        # Sine function
        x_sine = np.linspace(0, 128, 1000)
        y_sine = np.sin(x_sine + frame / 10)
        sine_line.set_data(x_sine, y_sine)

        return line, sine_line

    lineVal, sine_lineVal = update(0)
    
    
    # Create the animation
    # ani = FuncAnimation(fig, update, frames=range(len(generationData)), interval=200, blit=True)

    plt.show()




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