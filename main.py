import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 1,
    Seed = 0,
    UseLinReg = 0, # True for LinReg false for Sine
    UseCrowding = 0,
    UseReplacementSelection = 0,
    UseFitnessSelection = 0,
    BitstringLength = 5,
    MutationRate = 0.7,
    CrossoverRate = 0.7,
    PopSize = 100,
    AmountOfParents = 10
)