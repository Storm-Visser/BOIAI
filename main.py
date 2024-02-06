import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 100,
    Seed = 0,
    UseLinReg = 0, # True for LinReg false for Sine
    UseCrowding = 0,
    UseReplacementSelection = 0,
    UseFitnessSelection = 1,
    BitstringLength = 50,
    MutationRate = 0.1,
    CrossoverRate = 0.7,
    PopSize = 100,
    AmountOfParents = 50
)