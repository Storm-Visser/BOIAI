import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LinReg import LinReg

def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, UseReplacementSelection, UseFitnessSelection, BitstringLength, MutationRate, CrossoverRate, PopSize, AmountOfParents, Constraint, UseDeterministic):
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
        Children, ChildParentCombos = Crossover(Parents, CrossoverRate)
        # mutate some of the children
        ChildrenM, ChildParentCombosM = Mutate(Children, MutationRate, ChildParentCombos)
        # select survivors for next gen
        NewPop = PopFitness
        if UseCrowding: #survivor selection 3 Crowding stuff
            if UseDeterministic:
                NewPop = DeterministicCrowding(NewPop, ChildParentCombosM, UseLinReg, regressor, data, Seed, Constraint)
            else:
                pass
            # Still needed?
            Match(ChildrenM)
            Compete(ChildrenM)

        elif UseFitnessSelection: #survivor selection 1 Add x children to existing pop, select top fitness
            NewPop = FitnessSelection(PopFitness, ChildrenM, UseLinReg, PopSize, regressor, data, Seed, Constraint)
        elif UseReplacementSelection: #survivor selection 2 Replace bot X of population with the new children
            NewPop = ReplacementSelection(PopFitness, ChildrenM, UseLinReg, PopSize, regressor, data, Seed, Constraint)
        Pop = [x[0] for x in NewPop]
        Results.append(SaveResults(NewPop))
        #print(SaveResults(NewPop))
        PopEachGeneration.append(Pop)
    if not UseLinReg :
        CreateSineGraph(PopEachGeneration, Constraint)
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
    ChildrenParentCombo = []
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
        ChildrenParentCombo.extend([[Child1,Parent],[Child2, OtherParent]])
    return Children, ChildrenParentCombo

def Mutate(Children, Rate, ChildParentCombos):
    if len(Children) <= 0: raise Exception("Children array empty")
    returnChildren = []
    ChildParentCombosM = ChildParentCombos.copy()
    for Child in Children:
        newChild = Child
        if random.random() < Rate:
            # get a random place to mutate
            i = random.randint(0, len(Child) - 1)
            # Explode the child
            newChild = [*Child]
            # mutate part of the child
            newChild[i] = str((int(Child[i]) + 1) % 2)
            # Glue the child back together
            newChild = "".join(newChild)
        # add the new child
        returnChildren.append(newChild)
        for Combo in ChildParentCombosM:
            if Combo[0] == Child:
                Combo[0] = newChild
    return returnChildren, ChildParentCombosM

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

def DeterministicCrowding(Pop, ParentChildrenCombosM, UseLinReg, Regressor, Data, Seed, Constraint):
    NewPop = Pop.copy()
    # print(NewPop)
    for Combo in ParentChildrenCombosM:
        # Get the fitness of the parent and the child
        if UseLinReg:
            FitC = fitnessML(Regressor, Data, Combo[0], Seed)
            FitP = fitnessML(Regressor, Data, Combo[1], Seed)
        else: 
            FitC = errorSine(Combo[0], Constraint)
            FitP = errorSine(Combo[1], Constraint)
        # get probability of replacement, logic reversed bc minimization
        ProbabilityOfReplacement = 0.5
        if FitC < FitP:
            ProbabilityOfReplacement = 1.0
        elif FitC == FitP:
            ProbabilityOfReplacement = 0.5
        elif FitC > FitP :
            ProbabilityOfReplacement = 0.0
        # replace the parent with the child
        if random.random() < ProbabilityOfReplacement:
            # remove parent
            NewPop.remove((Combo[1], FitP))
            # Add child
            NewPop.append((Combo[0], FitC))
    return NewPop

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

def fitnessML(regressor: LinReg, data:pd.DataFrame, bitstring, Seed):
    # Create bitarray from bitstring
    BitArray = np.fromstring(bitstring,'u1') - ord('0')
    # Get the data corrosponding to the BitArray
    X = regressor.get_columns(data.values, BitArray)
    # Return the fitness
    return regressor.get_fitness(X[:,:-1], X[:,-1], Seed)

def errorSine(bitstring, Constraint):
    sin_value = fitnessSine(bitstring, Constraint)

    error = (2 - (sin_value + 1))/2

    return error

def fitnessSine(bitstring, constraint):
    bit_value = int(bitstring, 2) / (2 ** len(bitstring) - 1)

    x_value = bit_value * 128

    if x_value < constraint[0]:
        distance = constraint[0] - x_value
    elif x_value > constraint[1]:
        distance = x_value - constraint[1]
    else:
        distance = 0

    sin_value = np.sin(x_value)
    sin_value -= 0.05 * distance

    return sin_value

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

    ani = FuncAnimation(fig, update, frames=range(len(generationData)), interval=100, blit=False)

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