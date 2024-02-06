import numpy as np
import pandas as pd

from LinReg import LinReg


def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, BitstringLength, MutationRate, CrossoverRate):
    Pop = InitPop(BitstringLength)
    Results = []
    regressor = LinReg()
    data = pd.read_csv("dataset.txt", header=None)

    for _ in range(AmountOfGen):
        # where do we select fitness funtion?
        Selection(Pop)
        Crossover(Pop, CrossoverRate)
        Mutate(Pop, MutationRate)
        if UseCrowding:
            Match(Pop)
            Compete(Pop)
        Results.append(SaveResults())
    return


def InitPop(BitstringLength):
    return []

def Selection(Pop):
    return[]

def Crossover(Pop, Rate):
    return[]

def Mutate(Pop, Rate):
    return[]

def Match(Pop):
    return[]

def Compete(Pop):
    return[]

def SaveResults(Pop):
    return[]

def fitnesML(regressor: LinReg, data:pd.DataFrame, bitstring:str):
    X = regressor.get_columns(data.values, bitstring)
    return regressor.get_fitness(X[:,:-1], X[:,-1])

def fitnesIntValue(bitstring):
    return int(bitstring, 2)

