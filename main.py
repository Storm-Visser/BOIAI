import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 30,
    Seed = 10,
    UseLinReg = 0, # True for LinReg false for Sine
    UseCrowding = 1, #1: De Jong's Scheme, 2: 2nd scheme
    UseReplacementSelection = 0,
    UseFitnessSelection = 0,
    BitstringLength = 10, #101 for LinReg
    MutationRate = 0.5,
    CrossoverRate = 0.7,
    PopSize = 100,
    AmountOfParents = 10,
    Constraint = [5,10]
)