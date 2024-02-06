import numpy as np
import pandas as pd
import math
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LinReg import LinReg

def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, UseReplacementSelection, UseFitnessSelection, BitstringLength, MutationRate, CrossoverRate, PopSize, AmountOfParents, Constraint):
    Pop = InitPop(BitstringLength, PopSize)
    # print(Pop)
    Results = []
    regressor = LinReg()
    data = pd.read_csv("Data/Data.txt", header=None)
    PopEachGeneration = []

    for _ in range(AmountOfGen):
        #get the top X amount of parents
        Parents, PopFitness = Selection(Pop, UseLinReg, AmountOfParents, regressor, data, Seed, Constraint)
        # use crossover to create children
        Children = Crossover(Parents, CrossoverRate)
        # mutate some of the children
        ChildrenM = Mutate(Children, MutationRate)
        # select survivors for next gen
        NewPop = PopFitness
        if UseCrowding != 0: #survivor selection 3 Crowding stuff
            if UseCrowding == 1: # De Jong's scheme
                selected_individuals = RandomMatch(PopFitness, ChildrenM)
                NewPop = DeJongsCompete(PopFitness, selected_individuals)
            elif UseCrowding == 2:
                RandomMatch(ChildrenM)
                Compete(ChildrenM)
        elif UseFitnessSelection: #survivor selection 1 Add x children to existing pop, select top fitness
            NewPop = FitnessSelection(PopFitness, ChildrenM, UseLinReg, PopSize, regressor, data, Seed, Constraint)
        elif UseReplacementSelection: #survivor selection 2 Replace bot X of population with the new children
            NewPop = ReplacementSelection(PopFitness, ChildrenM, UseLinReg, PopSize, regressor, data, Seed, Constraint)
        Pop = [x[0] for x in NewPop]
        Results.append(SaveResults(NewPop))
        #print(SaveResults(NewPop))
        PopEachGeneration.append(Pop)
    #CreateGraph(Results) 
    CreateSineGraph(PopEachGeneration, Constraint)
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

def Selection(Pop, UseLinReg, AmountOfParents, Regressor, Data, Seed, Constraint):
    Selected = []
    for Individual in Pop:
        if UseLinReg:
            Fitness = fitnessML(Regressor, Data, Individual, Seed)
        else: 
            Fitness = errorSine(Individual, Constraint)
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

def FitnessSelection(Pop, Children, UseLinReg, PopSize, Regressor, Data, Seed, Constraint):
    # get the fitness of the children
    FitnessChildren = []
    for Child in Children:
        if UseLinReg:
            Fitness = fitnessML(Regressor, Data, Child, Seed)
        else: 
            Fitness = errorSine(Child, Constraint)
        #Add the individual and its value to the dict
        FitnessChildren.append((Child, Fitness))
    # add the dicts together
    TotalFitness = Pop + FitnessChildren
    # sort them by fitness
    SortedTotalFitness = sorted(TotalFitness, key=lambda x: x[1], reverse=False)
    # get the top amount of population
    NewPop = SortedTotalFitness[:PopSize]
    return NewPop

def ReplacementSelection(Pop, Children, UseLinReg, Regressor, Data, Seed, Constraint):
    childrenToAdd = []
    fitnessVal = None
    for Child in Children:
        if UseLinReg:
            fitnessVal = fitnessML(Regressor, Data, Child, Seed)
        else: 
            fitnessVal = errorSine(Child, Constraint)
        childrenToAdd.append((Child, fitnessVal))
        
    Pop[-len(Children):] = childrenToAdd
    return Pop

def DeJongsCompete(Pop, selected_individuals):

    for pair in selected_individuals:
        winner = [None, -(sys.maxsize - 1)]
        loser = [None, -(sys.maxsize - 1)]
        for Individual in pair:
            Individual_bitValue = scaledBitValue(Individual) / 128.0
            Cumulative_distance = 0.0
            for Neighbor in Pop:
                Neighbor_bitValue = scaledBitValue(Neighbor[0]) / 128.0
                Cumulative_distance += abs(Neighbor_bitValue - Individual_bitValue)
            
            Individual_fitness = errorSine(Individual, Constraint=None)
            Individual_fitness -= Cumulative_distance / len(Pop)

            if Individual_fitness > winner[1]:
                loser = winner
                winner = [Individual, Individual_fitness]
            else:
                loser = [Individual, Individual_fitness]

        if pair == (loser[0], winner[0]): #i.e. parent lost
            loser = [x for x in Pop if x[0] == loser[0]][0]
            winner = (winner[0], errorSine(winner[0], Constraint=None))
            Pop.remove(loser)
            Pop.append(winner)
   
    return Pop
        
def RandomMatch(Pop, Children):
    # Matches every child to a random individual in the population
    # Returns a list of [Parent, Child]
    selected_pairs = []
    Pop = [x[0] for x in Pop]
    selected_individuals = np.random.choice(Pop, size=len(Children), replace=False)
    for i, Child in enumerate(Children):
        selected_pairs.append((selected_individuals[i], Child))
    return selected_pairs

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

def fitnessML(regressor: LinReg, data:pd.DataFrame, bitstring, Seed):
    BitArray = np.fromstring(bitstring,'u1') - ord('0')
    X = regressor.get_columns(data.values, BitArray )
    return regressor.get_fitness(X[:,:-1], X[:,-1], Seed)

def errorSine(bitstring, Constraint):
    sin_value = fitnessSine(bitstring, Constraint)

    error = (2 - (sin_value + 1))/2

    return error

def fitnessSine(bitstring, constraint):
    x_value = scaledBitValue(bitstring)

    if constraint == None:
        distance = 0
    elif x_value < constraint[0]:
        distance = constraint[0] - x_value
    elif x_value > constraint[1]:
        distance = x_value - constraint[1]
    else:
        distance = 0

    sin_value = np.sin(x_value)
    sin_value -= 0.05 * distance

    return sin_value

def scaledBitValue(bitstring):
    bit_value = int(bitstring, 2) / (2 ** len(bitstring) - 1)

    return bit_value * 128

def CreateSineGraph(generationData, Constraint):

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='blue', marker='o', linestyle='', markersize=3, label='Individuals')
    sine_line, = ax.plot([], [], color='red', linewidth=0.5, alpha=0.5, label='Sine Function')
    ax.set_xlabel('Chromosome value')
    ax.set_ylabel('Fitness')
    title_text = ax.set_title('')
    ax.set_xlim(0, 128)
    ax.set_ylim(-5.2, 1.2)
    # ax.legend()

    def update(frame):
        x_values = [(int(x, 2) / (2 ** len(x) - 1)) * 128 for x in generationData[frame]]
        fitness_values = [fitnessSine(x, Constraint) for x in generationData[frame]]
        line.set_data(x_values, fitness_values)

        # print("x_values", x_values)
        # print("fitness_values", fitness_values)

        x_sine = np.linspace(0, 128, 1000)
        y_sine = np.sin(x_sine + len(generationData))
        sine_line.set_data(x_sine, y_sine)

        title_text.set_text(f'Generation: {frame}')

        return line, sine_line, title_text

    ani = FuncAnimation(fig, update, frames=range(len(generationData)), interval=500, blit=False)

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