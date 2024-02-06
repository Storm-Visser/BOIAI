


def StartSim(AmountOfGen, Seed, UseLinReg, UseCrowding, BitstringLength, MutationRate, CrossoverRate):
    Pop = InitPop(BitstringLength)
    Results = []
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




    ## mycomment