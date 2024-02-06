import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 20,
    Seed = 10,
    UseLinReg = 0, # True for LinReg false for Sine
    UseCrowding = 0,
    UseReplacementSelection = 1,
    UseFitnessSelection = 1,
    BitstringLength = 10, #101 for LinReg
    MutationRate = 0.5,
    CrossoverRate = 0.7,
    PopSize = 100,
    AmountOfParents = 10,
    Constraint = [5,10]
)