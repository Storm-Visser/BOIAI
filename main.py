import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 0,
    Seed = 0,
    UseLinReg = 0, # True for LinReg false for Sine
    UseCrowding = 0,
    BitstringLength = 15,
    MutationRate = 0.0,
    CrossoverRate = 0.0,
    PopSize = 100,
    AmountOfParents = 10
)