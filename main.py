import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 100,
    Seed = 0,
    UseLinReg = 1, # True for LinReg false for Sine
    UseCrowding = 0,
    UseReplacementSelection = 0,
    UseFitnessSelection = 1,
    BitstringLength = 91,
    MutationRate = 1,
    CrossoverRate = 0.8,
    PopSize = 100,
    AmountOfParents = 10
)